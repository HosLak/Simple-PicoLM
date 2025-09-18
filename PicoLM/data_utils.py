import torch
from torch.utils.data import Dataset
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List
import os
import pickle

from .config import ModelConfig

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set all seeds to {seed}")

def load_and_cache_data(config: ModelConfig, cache_dir: str = "data_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.max_tokens}.pkl"

    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size

        print(f"Loaded {len(texts)} Stories, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"Processing new data (will cache for future use)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Hosseinlack123/Blueberry-testtokenizer")

    # Load dataset
    dataset = load_dataset("Hosseinlack123/Blueberry-testdataset")['train']

    texts = []
    for i, item in enumerate(dataset):
        texts.append(item["text"])
    
    # Tokenize
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text+'\n\n\n', add_special_tokens=False)
        all_tokens.extend(tokens)

    all_tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(all_tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size

    # Cache the processed data
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': all_tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"Cached data to {cache_file}")
    return texts, tokenizer, all_tokens

class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 256, stride: int = 128):
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
