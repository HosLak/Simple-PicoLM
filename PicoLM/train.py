import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist
import time
import warnings
import os
import sys
warnings.filterwarnings('ignore')

from .config import ModelConfig
from .data_utils import set_seed, load_cached_data, TextTokenDataset
from .training_utils import train_model, save_model

def main():
    try:
        dist.init_process_group(backend="nccl")
    except:
        pass
    ddp = int(os.environ.get('RANK', -1)) != -1
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_master = rank == 0
    
    if is_master:
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            try:
                print(f'GPU {torch.cuda.get_device_name(local_rank)}')
                print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            except Exception:
                pass
        print(f"Available GPUs: {torch.cuda.device_count()}")

    # Set seed
    set_seed(1337)

    # Create config
    config = ModelConfig()
    if is_master:
        print(f"\nModel Configuration:")
        print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
        print(f"   Training: {config.max_steps} steps, batch size {config.batch_size}")
        print(f"   Data: {config.max_tokens:,} tokens, seq_len {config.max_seq_len}")

    # Load data
    tokens = load_cached_data(config)
    if tokens == None:
        sys.exit()
    train_dataset = tokens[:int(len(tokens) * 0.9)]
    val_dataset = tokens[int(len(tokens) * 0.9):]
    # dataset = TextTokenDataset(tokens, config.max_seq_len, config.stride)
        
    # Train/val split
    # val_size = len(tokens) // 10
    # train_size = len(tokens) - val_size
    # train_dataset, val_dataset = torch.utils.data.random_split(
    #     dataset, [train_size, val_size], generator=torch.Generator().manual_seed(1337)
    # )
    if is_master:
        num_gpus = torch.cuda.device_count()
        effective_batch_size = config.batch_size * config.gradient_accumulation_steps * config.max_seq_len * num_gpus
        print(f'Effective Batch Size: {effective_batch_size}, and for each gpu -> {effective_batch_size//num_gpus}')
        print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
        print()
        print(f'DDP: {ddp}')
        
    shuffle = True
    # if ddp:
    #     train_sampler = DistributedSampler(
    #         train_dataset,
    #         num_replicas=world_size,
    #         rank=rank,
    #         shuffle=True,
    #         drop_last=False
    #     )
        
    #     valid_sampler = DistributedSampler(
    #         val_dataset,
    #         num_replicas=world_size,
    #         rank=rank,
    #         shuffle=False,
    #         drop_last=False
    #     )
    #     shuffle = False
    # else:
    #     train_sampler = None
    #     valid_sampler = None
    #     shuffle = True
        
    # # Dataloader
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config.batch_size,
    #     shuffle=shuffle,
    #     sampler=train_sampler,
    #     num_workers=config.num_workers,
    #     # pin_memory=torch.cuda.is_available(),
    #     drop_last=True
    # )
    
    # valid_loader = DataLoader(
    #     val_dataset,
    #     batch_size=config.batch_size,
    #     shuffle=shuffle,
    #     sampler=valid_sampler,
    #     num_workers=config.num_workers,
    #     # pin_memory=torch.cuda.is_available(),
    #     drop_last=False
    # )
    
    # Train model
    if is_master:
        start_time = time.time()
    result = train_model(config, train_dataset, val_dataset, is_master, rank, world_size)
    
    if is_master:
        model, final_metrics = result
        total_time = time.time() - start_time
        print(f"\n TRAINING COMPLETED!")
        save_model(model.state_dict(), "PicoLMModel.pt")
        print(f" Total time: {total_time/60:.1f} minutes")
        print(f" Final Results:")
        # print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
        # print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
        # print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")
