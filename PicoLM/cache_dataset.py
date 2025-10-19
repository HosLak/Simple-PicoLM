from datasets import load_dataset
from tqdm import tqdm
import pickle

def cache(config):
    dataset_name = config.dataset_name.split('/')[-1]
    cache_prefix = f"{config.data_cache_dir}/tokenized_{dataset_name}_{config.max_tokens}"
    
    print(f"Processing new data (will cache for future use)")

    # Load tokenizer
    tokenizer = config.tokenizer
    
    # Load dataset
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    
    texts = []
    for i, item in enumerate(dataset):
        texts.append(item["text"])
    
    # Tokenize
    print("Tokenizing texts...")        
    all_tokens = []
    total_tokens = 0
    chunk_id = 0
    
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text + tokenizer.eos_token, add_special_tokens=False)
        all_tokens.extend(tokens)
        
        if config.max_tokens != -1 and len(all_tokens) >= config.max_tokens:
            break 
        
        while all(tokens) >= config.data_chunk_size:
            chunk_tokens = all_tokens[:config.data_chunk_size]
            all_tokens = all_tokens[config.data_chunk_size:]
            
            chunk_file = f'{cache_prefix}_chunk{chunk_id}.pkl'
            with open(chunk_file, 'wb') as f:
                pickle.dump({'tokens': chunk_file}, f)
                
            total_tokens += len(chunk_tokens)
            print(f"Saved chunk {chunk_id} with {len(chunk_tokens):,} tokens -> {chunk_file}")
            chunk_id += 1
            
        if config.max_tokens != -1 and total_tokens >= config.max_tokens:
            break
        
    if all_tokens:
        chunk_file = f"{cache_prefix}_chunk{chunk_id}.pkl"
        with open(chunk_file, "wb") as f:
            pickle.dump({"tokens": all_tokens}, f)

        print(f"Saved final chunk {chunk_id} with {len(all_tokens):,} tokens -> {chunk_file}")
        total_tokens += len(all_tokens)

    print(f"Finished caching {total_tokens:,} tokens in {chunk_id+1} chunks")