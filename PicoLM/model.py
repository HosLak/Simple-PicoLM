import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .data_utils import set_seed


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
        
        # gate_scores_raw = self.gate_linear(Q)
        # gate_scores = F.sigmoid(gate_scores_raw)
        # attn_output = attn_output * gate_scores
        
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

class PicoMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_layers: int, layer_idx: int, dropout: float = 0.1):
        super().__init__()

        self.d_ff = 16 * round(((1.5 + (4.0 - 1.5) * ((layer_idx - 1) / (n_layers - 1))) * d_model) / 16)

        self.w1 = nn.Linear(d_model, self.d_ff, bias=False) # up_proj
        self.w2 = nn.Linear(self.d_ff, d_model, bias=False) # down_proj
        self.w3 = nn.Linear(d_model, self.d_ff, bias=False) # gate_proj
        self.w3.zero_init = 1
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Gated Linear Unit (GLU)
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

        # UnTie weights
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # self.lm_head.weight = self.token_embedding.weight
        self.lm_head.zero_init = 1

        self.apply(self._init_weights)
        
        # Depth-aware init
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
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
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
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
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
