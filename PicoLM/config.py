from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    # Model architecture
    d_model: int = 256
    n_heads: int = 4
    n_kv_heads: int = field(init=False)
    n_layers: int = 8
    
    d_ff: int = field(init=False)
    batch_size: int = 16
    max_steps: int = 1500 # max_tokens // (batch_size * gradient_accumulation_steps * max_seq_len) = 1 epoch
    rope_theta: float = 100000.0

    # Training parameters
    gradient_accumulation_steps: int = 4 # 16 * 4 = 64
    muon_lr: float = 1e-2
    adamw_lr: float = 2e-3
    adamw_betas: tuple = (0.9, 0.95)

    # Data parameters
    max_seq_len: int = 256
    multiple_of: int = 128
    stride: int = field(init=False)
    max_tokens: int = -1 # -1 if you want to ues entire of dataset
    dataset_name: str = "Hosseinlack123/PicoLM-dataset"

    # Evaluation
    eval_every: int = 150
    eval_steps: int = 30

    # Regularization
    weight_decay: float = 0.01
    dropout: float = 0.0
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = self.d_model // self.n_heads
        
        # Set stride conditionally
        if self.max_tokens > 10000000:
            self.stride = self.max_seq_len // 2
        else:
            self.stride = self.max_seq_len
        
        # Set d_ff to 4 times d_model
        self.d_ff = int(self.multiple_of * int((((self.d_model * 4 * 2 / 3) * 1.3) + self.multiple_of + 1) // self.multiple_of))
        
        assert self.n_heads % 4 == 0, "n_heads must be divisible by 4"
        self.n_kv_heads = self.n_heads // 4
