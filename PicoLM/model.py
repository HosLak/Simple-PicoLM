import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .data_utils import set_seed


class PicoRotary(nn.Module):
    """RoPE for decoupled queries and keys in MLA"""
    def __init__(self, dim: int, max_seq_len: int, rope_theta: float = 10000.0):
        super().__init__()
        # Standard RoPE frequency computation
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # [seq_len, dim/2] -> [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos(), persistent=False)
        self.register_buffer('sin', emb.sin(), persistent=False)

    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        # x: [B, n_heads, T, head_dim]
        if seq_len is None:
            seq_len = x.size(-2)
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, T, dim]
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)
        x_float = x.float()
        return (x_float * cos + self._rotate_half(x_float) * sin).type_as(x)


class PicoMLA(nn.Module):
    """
    Multi-head Latent Attention (MLA) from DeepSeek-V2
    
    Key features:
    1. Low-rank KV compression to reduce KV cache
    2. Decoupled RoPE for positional encoding
    3. Low-rank Q compression for efficiency
    4. Gating mechanism (optional, from your original code)
    """
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        max_seq_len: int, 
        device,
        kv_lora_rank: int = None, 
        q_lora_rank: int = None, 
        qk_nope_head_dim: int = None,
        qk_rope_head_dim: int = None,
        v_head_dim: int = None,
        dropout: float = 0.1, 
        rope_theta: float = 10000.0,
        use_gating: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.use_gating = use_gating
        
        # Default MLA dimensions (following DeepSeek-V2 ratios)
        self.kv_lora_rank = kv_lora_rank or d_model // 4  # c_kv
        self.q_lora_rank = q_lora_rank  # c_q (None means no Q compression)
        self.qk_nope_head_dim = qk_nope_head_dim or d_model // n_heads // 2  # d_h^{nope}
        self.qk_rope_head_dim = qk_rope_head_dim or d_model // n_heads // 2  # d_h^{rope}
        self.v_head_dim = v_head_dim or d_model // n_heads  # d_h^v
        
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        
        if self.q_lora_rank is not None:
            self.q_down_proj = nn.Linear(d_model, self.q_lora_rank, bias=False)
            self.q_down_norm = nn.RMSNorm(self.q_lora_rank, eps=1e-6)
            self.q_up_proj = nn.Linear(
                self.q_lora_rank, 
                n_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), 
                bias=False
            )
        else:
            self.q_proj = nn.Linear(
                d_model, 
                n_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), 
                bias=False
            )
        
        self.kv_down_proj = nn.Linear(d_model, self.kv_lora_rank, bias=False)
        self.kv_down_norm = nn.RMSNorm(self.kv_lora_rank, eps=1e-6)
        
        self.k_up_proj = nn.Linear(
            self.kv_lora_rank, 
            n_heads * self.qk_nope_head_dim, 
            bias=False
        )
        
        self.v_up_proj = nn.Linear(
            self.kv_lora_rank, 
            n_heads * self.v_head_dim, 
            bias=False
        )
        
        self.k_rope_proj = nn.Linear(d_model, self.qk_rope_head_dim, bias=False)
        
        self.rotary = PicoRotary(self.qk_rope_head_dim, max_seq_len, rope_theta)
        
        self.w_o = nn.Linear(n_heads * self.v_head_dim, d_model, bias=False)
        self.w_o.zero_init = True
        
        if self.use_gating:
            self.gate_proj = nn.Linear(d_model, n_heads * self.v_head_dim, bias=False)
        
        self.q_norm = nn.RMSNorm(self.qk_head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.qk_head_dim, eps=1e-6)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)

    def forward(self, x: torch.Tensor, kv_cache: dict = None, use_cache: bool = False):
        """
        Args:
            x: [B, T, D] input tensor
            kv_cache: Optional dict containing cached KV states
            use_cache: Whether to return updated cache
        
        Returns:
            output: [B, T, D]
            (optional) new_kv_cache: dict with compressed KV cache
        """
        B, T, D = x.shape
        
        if self.q_lora_rank is not None:
            q_compressed = self.q_down_proj(x)  # [B, T, c_q]
            q_compressed = self.q_down_norm(q_compressed)
            q = self.q_up_proj(q_compressed)  # [B, T, n_heads * qk_head_dim]
        else:
            q = self.q_proj(x)  # [B, T, n_heads * qk_head_dim]
        
        # Reshape Q: [B, T, n_heads, qk_head_dim] -> [B, n_heads, T, qk_head_dim]
        q = q.view(B, T, self.n_heads, self.qk_head_dim).transpose(1, 2)
        
        # Split Q into nope and rope parts
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        c_kv = self.kv_down_proj(x)  # [B, T, c_kv] - THIS IS WHAT WE CACHE!
        c_kv = self.kv_down_norm(c_kv)
        
        # Compute decoupled RoPE key
        k_rope = self.k_rope_proj(x)  # [B, T, qk_rope_head_dim]
        
        # Handle KV cache for inference
        if kv_cache is not None:
            # Concatenate with cached compressed KV
            c_kv = torch.cat([kv_cache['c_kv'], c_kv], dim=1)
            k_rope = torch.cat([kv_cache['k_rope'], k_rope], dim=1)
        
        new_kv_cache = None
        if use_cache:
            new_kv_cache = {
                'c_kv': c_kv,      # Only cache the compressed representation!
                'k_rope': k_rope,  # And the rope keys
            }
        
        S = c_kv.size(1)
        
        k_nope = self.k_up_proj(c_kv)  # [B, S, n_heads * qk_nope_head_dim]
        k_nope = k_nope.view(B, S, self.n_heads, self.qk_nope_head_dim).transpose(1, 2)
        
        # V
        v = self.v_up_proj(c_kv)  # [B, S, n_heads * v_head_dim]
        v = v.view(B, S, self.n_heads, self.v_head_dim).transpose(1, 2)
        
        # K rope part - expand to all heads
        k_rope = k_rope.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # [B, n_heads, S, qk_rope_head_dim]
        
        # Only apply RoPE to the rope parts
        q_rope = self.rotary(q_rope, seq_len=T)
        # For k_rope, we need to handle the full sequence (including cache)
        k_rope = self.rotary(k_rope, seq_len=S)
        
        # Combine nope and rope parts
        q = torch.cat([q_nope, q_rope], dim=-1)  # [B, n_heads, T, qk_head_dim]
        k = torch.cat([k_nope, k_rope], dim=-1)  # [B, n_heads, S, qk_head_dim]
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=(kv_cache is None),  # Only causal mask during training
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale
        )  # [B, n_heads, T, v_head_dim]
        
        if self.use_gating:
            gate = self.gate_proj(x)  # [B, T, n_heads * v_head_dim]
            gate = gate.view(B, T, self.n_heads, self.v_head_dim).transpose(1, 2)
            attn_output = attn_output * F.silu(gate)
        
        attn_output = attn_output.transpose(1, 2).reshape(B, T, -1)  # [B, T, n_heads * v_head_dim]
        output = self.w_o(attn_output)
        
        if use_cache:
            return output, new_kv_cache
        return output


class PicoMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_layers: int, layer_idx: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_ff = d_ff
        # self.d_ff = 16 * round(((1.5 + (4.0 - 1.5) * ((layer_idx - 1) / (n_layers - 1))) * d_model) / 16)

        self.w1 = nn.Linear(d_model, self.d_ff, bias=False)  # up_proj
        self.w2 = nn.Linear(self.d_ff, d_model, bias=False)  # down_proj
        self.w3 = nn.Linear(d_model, self.d_ff, bias=False)  # gate_proj
        self.w2.zero_init = True
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class PicoBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int, 
        max_seq_len: int, 
        device, 
        n_layers: int, 
        layer_idx: int, 
        dropout: float = 0.1, 
        rope_theta: float = 10000.0,
        kv_lora_rank: int = None,
        q_lora_rank: int = None,
        qk_nope_head_dim: int = None,
        qk_rope_head_dim: int = None,
        v_head_dim: int = None,
        use_gating: bool = True,
    ):
        super().__init__()
        
        self.attention = PicoMLA(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            device=device,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            dropout=dropout,
            rope_theta=rope_theta,
            use_gating=use_gating,
        )
        
        self.feed_forward = PicoMLP(d_model, d_ff, n_layers, layer_idx, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None, use_cache=False):
        # Pre-norm architecture with RMSNorm
        normed_x = F.rms_norm(x, (x.size(-1),))
        
        if use_cache:
            attn_out, new_cache = self.attention(normed_x, kv_cache, use_cache=True)
        else:
            attn_out = self.attention(normed_x)
            new_cache = None
        
        attn_out = F.rms_norm(attn_out, (attn_out.size(-1),))
        x = x + self.dropout(attn_out)
        
        ff_out = F.rms_norm(self.feed_forward(F.rms_norm(x, (x.size(-1),))), (x.size(-1),))
        x = x + self.dropout(ff_out)
        
        if use_cache:
            return x, new_cache
        return x


class PicoLM(nn.Module):
    def __init__(self, config: ModelConfig, device: torch.device):
        super().__init__()
        set_seed(1337)
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        self.transformer_blocks = nn.ModuleList([
            PicoBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                device=device,
                n_layers=config.n_layers,
                layer_idx=layer_idx,
                dropout=config.dropout,
                rope_theta=config.rope_theta,
                kv_lora_rank=config.kv_lora_rank,
                q_lora_rank=config.q_lora_rank,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                use_gating=config.use_gating,
            )
            for layer_idx in range(config.n_layers)
        ])

        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # self.lm_head.weight = self.token_embedding.weight
        self.lm_head.zero_init = True

        self.apply(self._init_weights)
        
        # Depth-aware init
        for i, block in enumerate(self.transformer_blocks):
            std = 0.04 - (i / (config.n_layers - 1)) * 0.04 if config.n_layers > 1 else 0.02
            torch.nn.init.normal_(block.feed_forward.w1.weight, mean=0.0, std=std)
            torch.nn.init.normal_(block.feed_forward.w2.weight, mean=0.0, std=std)

    def _init_weights(self, module):
        torch.manual_seed(1337)
        
        if hasattr(module, 'zero_init') and module.zero_init:
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=1e-8) 
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, kv_cache=None, use_cache=False):
        """
        Args:
            x: [B, T] input token ids
            kv_cache: List of layer caches for inference
            use_cache: Whether to return updated cache
        """
        x = self.token_embedding(x)
        x = self.position_dropout(x)

        new_kv_cache = [] if use_cache else None
        
        for i, block in enumerate(self.transformer_blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            
            if use_cache:
                x, new_layer_cache = block(x, layer_cache, use_cache=True)
                new_kv_cache.append(new_layer_cache)
            else:
                x = block(x)

        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        
        if hasattr(self.config, 'softcap') and self.config.softcap is not None:
            logits = self.config.softcap * torch.tanh(logits / self.config.softcap)
        
        if use_cache:
            return logits, new_kv_cache
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate tokens with KV cache for efficient inference"""
        kv_cache = None
        
        for _ in range(max_new_tokens):
            # Only process the last token if we have cache
            if kv_cache is not None:
                idx_cond = idx[:, -1:]
            else:
                idx_cond = idx
            
            logits, kv_cache = self.forward(idx_cond, kv_cache=kv_cache, use_cache=True)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


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