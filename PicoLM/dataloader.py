import os
import glob
import time
from collections import deque
import torch
import pyarrow.parquet as pq

class StreamingDataLoader:
    def __init__(self, data_dir, tokenizer, batch_size, block_size, split="train", device="cuda"):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.block_size = block_size 
        self.device = device
        
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        
        if split == "train":
            self.files = all_files[:-1]
        else:
            self.files = all_files[-1:]
            
        assert len(self.files) > 0, f"No parquet files found in {data_dir}"
        print(f"Initialized {split} loader with {len(self.files)} files.")

    def _document_generator(self, resume_state=None):
        pq_idx = 0
        start_rg_idx = 0
        
        if resume_state:
            pq_idx = resume_state.get("pq_idx", 0)
            start_rg_idx = resume_state.get("rg_idx", 0)

        while True: 
            if pq_idx >= len(self.files):
                pq_idx = 0 

            filepath = self.files[pq_idx]
            pf = pq.ParquetFile(filepath)
            
            rg_idx = start_rg_idx if start_rg_idx is not None else 0
            start_rg_idx = None 

            while rg_idx < pf.num_row_groups:
                table = pf.read_row_group(rg_idx)
                text_batch = table.column('text').to_pylist()
                
                yield text_batch, (pq_idx, rg_idx)
                
                rg_idx += 1 
            
            pq_idx += 1 

    def __iter__(self):
        token_buffer = deque()
        needed_tokens = self.batch_size * self.block_size + 1 
        
        bos_token = getattr(self.tokenizer, "bos_token_id", None)
        
        generator = self._document_generator()
        
        while True:
            while len(token_buffer) < needed_tokens:
                try:
                    text_batch, (pq_idx, rg_idx) = next(generator)
                except StopIteration:
                    break 
                
                for text in text_batch:
                    tokens = self.tokenizer.encode(text, add_special_tokens=True)
                    token_buffer.extend(tokens)
            
            if len(token_buffer) >= needed_tokens:
                batch_tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
                
                data_tensor = torch.tensor(batch_tokens, dtype=torch.long).pin_memory()
                
                x = data_tensor[:-1].view(self.batch_size, self.block_size)
                y = data_tensor[1:].view(self.batch_size, self.block_size)
                
                if self.device == "cuda":
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                
                yield x, y, {"pq_idx": pq_idx, "rg_idx": rg_idx}