from dataclasses import dataclass, field
from transformers import AutoTokenizer
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
import pickle
import time
import warnings
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed

warnings.filterwarnings('ignore')

@dataclass
class ModelConfig:
    # Model architecture
    d_model: int = 256
    n_heads: int = 4
    n_kv_heads: int = field(init=False)
    n_layers: int = 8
    d_ff: int = field(init=False)
    
    rope_theta: float = 100000.0

    # Training parameters
    batch_size: int = 16
    gradient_accumulation_steps: int = 4 # 16 * 4 = 64
    max_steps: int = 1500
    muon_lr: float = 1e-2
    adamw_lr: float = 2e-3
    adamw_betas: tuple = (0.9, 0.95)

    # Data parameters
    max_seq_len: int = 256
    multiple_of: int = 128
    stride: int = field(init=False)
    max_tokens: int = -1
    dataset_name: str = "Hosseinlack123/PicoLM-dataset"
    tokenizer_name: str = "Hosseinlack123/PicoLM-tokenizer"
    data_cache_dir: str = "data_cache"
    data_chunk_size: int = 1e7
    dataset_split: str = 'train'

    # Evaluation
    eval_every: int = 150
    eval_steps: int = 30

    # Regularization
    weight_decay: float = 0.01
    dropout: float = 0.0
    grad_clip: float = 1.0

    # Technical
    num_workers: int = 2
    use_amp: bool = True
    tokenizer: AutoTokenizer = field(init=False, repr=False)
    vocab_size: Optional[int] = None

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = self.d_model // self.n_heads
        
        if self.max_tokens > 10000000:
            self.stride = self.max_seq_len // 2
        else:
            self.stride = self.max_seq_len
        
        self.d_ff = int(self.multiple_of * int((((self.d_model * 4 * 2 / 3) * 1.3) + self.multiple_of + 1) // self.multiple_of))
        
        assert self.n_heads % 4 == 0, "n_heads must be divisible by 4"
        self.n_kv_heads = self.n_heads // 4
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        if self.vocab_size is None:
            self.vocab_size = self.tokenizer.vocab_size
        else:
            if self.vocab_size != len(self.tokenizer):
                print(f"Warning: provided vocab_size ({self.vocab_size}) != tokenizer vocab size ({len(self.tokenizer)})")


class PicoRotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, rope_theta: float = 10000.0):
        super().__init__()
        angular_freq = (1 / rope_theta) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class PicoAttn(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, max_seq_len: int, dropout: float = 0.1, rope_theta: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.repeats = self.n_heads // self.n_kv_heads

        total_qkv_dim = (self.n_heads + (2 * self.n_kv_heads)) * self.d_k

        self.qkv = nn.Linear(d_model, total_qkv_dim, bias=True)
        self.gate_linear = nn.Linear(self.d_k, self.d_k, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.w_o.zero_init = 1
        
        self.q_norm = nn.RMSNorm(self.d_k, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.d_k, eps=1e-6)
        
        self.rotary = PicoRotary(self.d_k, max_seq_len, rope_theta)
        self.dropout = dropout
        
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        qkv = self.qkv(x)
        q_size = self.n_heads * self.d_k
        kv_size = self.n_kv_heads * self.d_k
    
        Q, K, V = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
    
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.n_kv_heads, self.d_k).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.n_kv_heads, self.d_k).permute(0, 2, 1, 3)

        Q = self.rotary(self.q_norm(Q))
        K = self.rotary(self.k_norm(K))
        
        K = K.repeat_interleave(self.repeats, dim=1)
        V = V.repeat_interleave(self.repeats, dim=1)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        
        gate_scores_raw = self.gate_linear(Q)
        gate_scores = F.sigmoid(gate_scores_raw)
        attn_output = attn_output * gate_scores
        
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)


class PicoMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_layers: int, layer_idx: int, dropout: float = 0.1):
        super().__init__()

        self.d_ff = 16 * round(((1.5 + (4.0 - 1.5) * ((layer_idx - 1) / (n_layers - 1))) * d_model) / 16)

        self.w1 = nn.Linear(d_model, self.d_ff, bias=False)
        self.w2 = nn.Linear(self.d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, self.d_ff, bias=False)
        self.w3.zero_init = 1
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class PicoBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, max_seq_len: int, n_layers: int, layer_idx: int, dropout: float = 0.1, rope_theta: float = 10000.0):
        super().__init__()
        
        self.input_norm = nn.RMSNorm(d_model)
        self.attention = PicoAttn(d_model, n_heads, n_kv_heads, max_seq_len, dropout, rope_theta)
        self.post_attention_norm = nn.RMSNorm(d_model)
        
        self.pre_feedforward_norm = nn.RMSNorm(d_model)
        self.feed_forward = PicoMLP(d_model, d_ff, n_layers, layer_idx, dropout)
        self.post_feedforward_norm = nn.RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.post_attention_norm(self.attention(self.input_norm(x)))
        x = x + self.dropout(attn_out)
        
        ff_out = self.post_feedforward_norm(self.feed_forward(self.pre_feedforward_norm(x)))
        
        return x + self.dropout(ff_out)


class PicoLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        set_seed(1337)
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        self.transformer_blocks = nn.ModuleList([
            PicoBlock(config.d_model, config.n_heads, config.n_kv_heads, config.d_ff, config.max_seq_len, config.n_layers, layer_idx, config.dropout, config.rope_theta)
            for layer_idx in range(config.n_layers)
        ])

        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.zero_init = 1

        self.apply(self._init_weights)
        
        for i, block in enumerate(self.transformer_blocks):
            std = 0.04 - (i / (config.n_layers - 1)) * 0.04 if config.n_layers > 1 else 0.02
            torch.nn.init.normal_(block.feed_forward.w1.weight, mean=0.0, std=std)
            torch.nn.init.normal_(block.feed_forward.w2.weight, mean=0.0, std=std)

    def _init_weights(self, module):
        torch.manual_seed(1337)
        
        if hasattr(module, 'zero_init'):
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.weight) 
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)


def set_seed(seed: int = 1337, log: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if log:
        print(f"Set all seeds to {seed}")


def load_cached_data(config: ModelConfig, chunk_id: int = None):
    os.makedirs(config.data_cache_dir, exist_ok=True)
    dataset_name = config.dataset_name.split('/')[-1]
    cache_prefix = f"{config.data_cache_dir}/tokenized_{dataset_name}_{config.max_tokens}"

    chunk_files = sorted(
        [f for f in os.listdir(config.data_cache_dir) if f.startswith(os.path.basename(cache_prefix))],
        key=lambda x: int(x.split("chunk")[-1].split(".")[0])
    )

    if not chunk_files:
        print("No cached chunks found.")
        return None

    if chunk_id is not None:
        if chunk_id >= len(chunk_files):
            print(f"Requested chunk {chunk_id} but only {len(chunk_files)} chunks exist.")
            return None

        chunk_file = os.path.join(config.data_cache_dir, chunk_files[chunk_id])
        print(f"Loading cached data from {chunk_file}")
        with open(chunk_file, "rb") as f:
            cached_data = pickle.load(f)

        tokens = cached_data["tokens"]
        print(f"Loaded {len(tokens):,} tokens from chunk {chunk_id}")
        return tokens
    else:
        all_tokens = []
        for i, cf in enumerate(chunk_files):
            chunk_file = os.path.join(config.data_cache_dir, cf)
            with open(chunk_file, "rb") as f:
                cached_data = pickle.load(f)
            tokens = cached_data["tokens"]
            all_tokens.extend(tokens)
            print(f"Loaded {len(tokens):,} tokens from {cf}")

        print(f"Loaded total {len(all_tokens):,} tokens from {len(chunk_files)} chunks")
        return all_tokens


def setup_muon_optimizer(model: nn.Module, config: ModelConfig):
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.adamw_lr, weight_decay=config.weight_decay, betas=config.adamw_betas)

    return [muon_optimizer, adamw_optimizer]


def save_model(model, filepath="PicoLMModel.pt"):
    torch.save(model.state_dict(), filepath)
    print(f" Model weights saved to {filepath}")


class DataLoader:
    def __init__(self, tokens, batch_size, seq_len, process_rank, num_process, split):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_process = num_process
                
        total_tokens = len(tokens)
        total_per_process = total_tokens // num_process
        start_idx = process_rank * total_per_process
        
        if process_rank == num_process - 1:
            end_idx = total_tokens
        else: 
            end_idx = start_idx + total_per_process
            
        self.tokens = torch.tensor(tokens[start_idx:end_idx])
        self.current_position = 0
        
        print(f"Rank {process_rank}: Loaded {len(self.tokens):,} tokens (from {start_idx} to {end_idx})") 
        
    def next_batch(self):
        batch_size, seq_len = self.batch_size, self.seq_len
        needed_tokens = batch_size * seq_len + 1
        
        if self.current_position + needed_tokens >= len(self.tokens):
            self.current_position = 0
        
        buf = self.tokens[self.current_position:self.current_position + needed_tokens]
        
        x = (buf[:-1]).view(batch_size, seq_len)
        y = buf[1:].view(batch_size, seq_len)
        
        self.current_position += batch_size * seq_len
        
        return x, y


# ============= MAIN TRAINING CODE WITH ACCELERATE =============

# Initialize Accelerator
accelerator = Accelerator(
    gradient_accumulation_steps=4,
    mixed_precision='fp16' if True else 'no',  # می‌تونید 'bf16' هم استفاده کنید
    log_with=None,
)

# Set seed
set_seed(1337)
accelerate_set_seed(1337)

# Create config
config = ModelConfig()

if accelerator.is_main_process:
    print(f"\n{'='*60}")
    print(f"🚀 Training with Accelerate")
    print(f"{'='*60}")
    print(f"Device: {accelerator.device}")
    print(f"Number of processes: {accelerator.num_processes}")
    print(f"Mixed precision: {accelerator.mixed_precision}")
    print(f"\nModel Configuration:")
    print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
    print(f"   Training: {config.max_steps} steps, batch size {config.batch_size}")
    print(f"   Data: {config.max_tokens:,} tokens, seq_len {config.max_seq_len}")

# Load data
tokens = load_cached_data(config)
if tokens is None:
    if accelerator.is_main_process:
        print('No cached data found, please run run_cache_dataset.py first')
    exit()

train_dataset = tokens[:int(len(tokens) * 0.9)]
val_dataset = tokens[int(len(tokens) * 0.9):]

if accelerator.is_main_process:
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps * config.max_seq_len * accelerator.num_processes
    print(f'\nEffective Batch Size: {effective_batch_size:,}')
    print(f"Per GPU Batch Size: {effective_batch_size // accelerator.num_processes:,}")
    print(f"Dataset: {len(train_dataset):,} train, {len(val_dataset):,} val tokens")
    print(f"\n🔧 Training Small model with Muon optimizer\n")

# Initialize model
model = PicoLM(config)

if accelerator.is_main_process:
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

# Setup optimizers
optimizers = setup_muon_optimizer(model, config)

# Learning rate schedulers
schedulers = []
for optimizer in optimizers:
    effective_max_steps = config.max_steps // config.gradient_accumulation_steps
    warmup_steps = int(effective_max_steps * 0.06)
    milestone1 = int(effective_max_steps * 0.8)
    milestone2 = int(effective_max_steps * 0.9)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        elif step < milestone1:
            return 1.0
        elif step < milestone2:
            return 0.316
        else:
            return 0.1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    schedulers.append(scheduler)

# Prepare everything with accelerator
model, *optimizers = accelerator.prepare(model, *optimizers)

# Data loaders
train_loader = DataLoader(
    train_dataset, 
    config.batch_size, 
    config.max_seq_len, 
    accelerator.process_index, 
    accelerator.num_processes, 
    "train"
)

val_loader = DataLoader(
    val_dataset, 
    config.batch_size, 
    config.max_seq_len, 
    accelerator.process_index, 
    accelerator.num_processes, 
    "val"
)

# Training loop
model.train()
start_time = time.time()

if accelerator.is_main_process:
    pbar = tqdm(total=config.max_steps, desc="Training")

for step in range(config.max_steps):
    # Validation
    if step % 250 == 0:
        model.eval()
        val_loss_accum = 0.0
        val_loss_steps = 20
        
        with torch.no_grad():
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(accelerator.device), y.to(accelerator.device)
                
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                val_loss_accum += loss / val_loss_steps
        
        # Gather validation loss from all processes
        val_loss_accum = accelerator.gather(val_loss_accum).mean()
        
        if accelerator.is_main_process:
            print(f"\nValidation loss: {val_loss_accum.item():.4f}")
        
        model.train()
    
    # Training step
    t0 = time.time()
    
    with accelerator.accumulate(model):
        loss_accum = 0.0
        
        for micro_step in range(config.gradient_accumulation_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(accelerator.device), y.to(accelerator.device)
            
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            loss = loss / config.gradient_accumulation_steps
            
            # Accelerate handles backward automatically
            accelerator.backward(loss)
            loss_accum += loss.detach()
        
        # Gather loss from all processes
        loss_accum = accelerator.gather(loss_accum).mean()
        
        # Gradient clipping
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # Optimizer step
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()
    
    # Scheduler step
    for scheduler in schedulers:
        scheduler.step()
    
    # Logging
    if step % 5 == 0 and accelerator.is_main_process:
        perplexity = math.exp(min(loss_accum.item(), 20.0))
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        
        pbar.set_postfix({
            'loss': f'{loss_accum.item():.4f}',
            'dt': f'{dt*1000:.2f}ms',
            'ppl': f'{perplexity:.1f}',
            'muon_lr': f'{optimizers[0].param_groups[0]["lr"]:.2e}',
            'adamw_lr': f'{optimizers[1].param_groups[0]["lr"]:.2e}'
        })
        pbar.update(5)

# Wait for all processes
accelerator.wait_for_everyone()

# Save model
if accelerator.is_main_process:
    pbar.close()
    training_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"   Training time: {training_time/60:.1f} minutes")
    print(f"   Final train loss: {loss_accum.item():.4f}")
    
    # Unwrap model before saving
    unwrapped_model = accelerator.unwrap_model(model)
    save_model(unwrapped_model, "PicoLMModel.pt")
    print(f"\n🎉 Done!")