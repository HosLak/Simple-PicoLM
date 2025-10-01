import torch
from torch.utils.data import Dataset
import random
import numpy as np
from typing import List
import os
import pickle

from .config import ModelConfig

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

class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int, stride: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride if stride else seq_len
        
        self.num_samples = (len(self.tokens) - seq_len) // self.stride

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx = idx * self.stride
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y