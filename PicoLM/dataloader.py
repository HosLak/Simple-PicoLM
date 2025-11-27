import os
import glob
from collections import deque
from threading import Thread
from queue import Queue
import torch
import pyarrow.parquet as pq

class StreamingDataLoader:
    def __init__(self, data_dir, tokenizer, batch_size, block_size, split="train", device="cuda", prefetch_batches=3):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.block_size = block_size 
        self.device = device
        self.prefetch_batches = prefetch_batches
        
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        
        if split == "train":
            self.files = all_files[:-1]
        else:
            self.files = all_files[-1:]
            
        assert len(self.files) > 0, f"No parquet files found in {data_dir}"
        print(f"Initialized {split} loader with {len(self.files)} files.")

    def _document_generator(self, num_epochs=1):
        for epoch in range(num_epochs):
            for pq_idx, filepath in enumerate(self.files):
                pf = pq.ParquetFile(filepath)
                
                for rg_idx in range(pf.num_row_groups):
                    table = pf.read_row_group(rg_idx)
                    text_batch = table.column('text').to_pylist()
                    
                    tokens_batch = self.tokenizer(
                        text_batch, 
                        add_special_tokens=True,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                    )['input_ids']
                    
                    yield tokens_batch, (epoch, pq_idx, rg_idx)

    def _batch_generator(self):
        token_buffer = []
        needed_tokens = self.batch_size * self.block_size + 1
        
        for tokens_batch, state in self._document_generator():
            for tokens in tokens_batch:
                token_buffer.extend(tokens)
                
                while len(token_buffer) >= needed_tokens:
                    # Slice به جای popleft
                    batch_tokens = token_buffer[:needed_tokens]
                    token_buffer = token_buffer[needed_tokens:]
                    
                    data_tensor = torch.tensor(batch_tokens, dtype=torch.long)
                    
                    x = data_tensor[:-1].view(self.batch_size, self.block_size)
                    y = data_tensor[1:].view(self.batch_size, self.block_size)
                    
                    yield x, y, state

    def _prefetch_worker(self, queue: Queue):
        for x, y, state in self._batch_generator():
            if self.device == "cuda":
                x = x.pin_memory()
                y = y.pin_memory()
            queue.put((x, y, state))
        queue.put(None)  # Signal end

    def __iter__(self):
        queue = Queue(maxsize=self.prefetch_batches)
        worker = Thread(target=self._prefetch_worker, args=(queue,), daemon=True)
        worker.start()
        
        while True:
            item = queue.get()
            if item is None:
                break
            x, y, state = item
            if self.device == "cuda":
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
            yield x, y, state