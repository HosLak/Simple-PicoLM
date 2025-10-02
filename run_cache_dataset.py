# if you want to run on multiple GPUs, you need to run this code
import os
from PicoLM.config import ModelConfig
import PicoLM.cache_dataset as cache_dataset

if __name__ == "__main__":
    config = ModelConfig()
    os.makedirs(config.data_cache_dir, exist_ok=True)
    dataset_name = config.dataset_name.split('/')[-1]
    cache_prefix = f"{config.data_cache_dir}/tokenized_{dataset_name}_{config.max_tokens}"
    existing_chunks = [f for f in os.listdir(config.data_cache_dir) if f.startswith(os.path.basename(cache_prefix))]
    
    if len(existing_chunks) > 0:
        print(f"Chunks already exist in {config.data_cache_dir}")
    else:
        cache_dataset.cache(config)