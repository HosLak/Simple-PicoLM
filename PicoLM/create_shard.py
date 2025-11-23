import os
import time
import glob
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa
from datasets import load_dataset
from .config import ModelConfig


def check_existing_shards(output_dir):
    shard_files = glob.glob(os.path.join(output_dir, "shard_*.parquet"))
    return len(shard_files) > 0

def create_shards():
    config = ModelConfig()
    output_dir = config.dataset_cache_path

    ds = load_dataset(config.dataset_name, split='train', streaming=True)

    os.makedirs(output_dir, exist_ok=True)

    chars_per_shard = config.chars_per_shard
    row_group_size = config.row_group_size
    shard_docs = []
    shard_index = 0
    shard_characters = 0
    total_docs_processed = 0
    total_docs_characters = 0
    t0 = time.time()

    for doc in tqdm(ds):
        text = doc['text']
        shard_docs.append(text)
        shard_characters += len(text)
        total_docs_characters += len(text)
        if total_docs_characters // 4 >= config.max_tokens:
            break

        collected_enough_chars = shard_characters >= chars_per_shard
        docs_multiple_of_row_group_size = len(shard_docs) % row_group_size == 0

        if collected_enough_chars and docs_multiple_of_row_group_size:
            shard_path = os.path.join(output_dir, f"shard_{shard_index:04d}.parquet")
            shard_table = pa.Table.from_pydict({"text": shard_docs})

            pq.write_table(
                shard_table,
                shard_path,
                row_group_size=row_group_size,
                use_dictionary=False,
                compression="zstd",
                compression_level=3,
                write_statistics=False,
            )

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            total_docs_processed += len(shard_docs)

            print(f"Wrote {shard_path}. #documents: {len(shard_docs)} | #characters: {shard_characters} | time: {dt:.2f}s")

            shard_docs = []
            shard_characters = 0
            shard_index += 1

    if shard_docs:
        print("Writing remaining documents in the last shard...")
        shard_path = os.path.join(output_dir, f"shard_{shard_index:04d}.parquet")
        shard_table = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(
            shard_table,
            shard_path,
            row_group_size=row_group_size,
            use_dictionary=False,
            compression="zstd",
            compression_level=3,
            write_statistics=False,
        )
        print(f"Wrote final {shard_path}. #documents: {len(shard_docs)}")

    print(f"Sharding process finished for the {' '} downloaded shards.")

def start_sharding_process():
    config = ModelConfig()
    output_dir = config.dataset_cache_path

    if check_existing_shards(output_dir):
        print(f"Shards already exist in {output_dir}. Skipping sharding.")
    else:
        print(f"No existing shards found in {output_dir}. Starting sharding process...")
        create_shards()