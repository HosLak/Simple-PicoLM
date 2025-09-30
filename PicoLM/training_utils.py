import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import math
import time
import sys
from tqdm import tqdm

from .config import ModelConfig
from .model import PicoLM, Muon
from .data_utils import set_seed

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            x, y = x.to(device), y.to(device)

            with autocast(enabled=config.use_amp):
                # DataParallel
                if hasattr(model, 'module'):
                    logits = model.module(x)
                else:
                    logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

def setup_muon_optimizer(model: nn.Module, config: ModelConfig):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=0.95)
    adamw_optimizer = torch.optim.AdamW(adamw_params, lr=config.adamw_lr, weight_decay=config.weight_decay, betas=config.adamw_betas)

    return [muon_optimizer, adamw_optimizer]

def save_model(model, filepath="PicoLMModel.pt"):
    """Save only model weights (for inference)"""
    torch.save(model.state_dict(), filepath)
    print(f" Model weights saved to {filepath}")

def train_model(config: ModelConfig, train_loader: Dataset, val_loader: Dataset, is_master: bool = True):
    """Train the model with Muon optimizer"""
    if is_master:
        print(f"\n Training Small model with Muon optimizer")

    # Initialize model
    set_seed(1337)
    model = PicoLM(config)
    
    num_gpus = torch.cuda.device_count()
    ddp = int(os.environ.get('RANK', -1)) != -1

    if ddp:
        if is_master:
            print(f"Using {num_gpus} GPUs for DDP.")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
    else:
        if is_master:
            print("Using single GPU or CPU.")
        ddp_local_rank = 0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        
    model = torch.compile(model)
    raw_model = model.module if ddp else model
    
    if is_master:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        total_batch_size = config.batch_size * config.gradient_accumulation_steps * config.max_seq_len * num_gpus
        print(f"total batch size: {total_batch_size}")

    # Setup optimizers
    optimizers = setup_muon_optimizer(raw_model, config)

    # Learning rate schedule
    schedulers = []
    for optimizer in optimizers:
        effective_max_steps = config.max_steps // config.gradient_accumulation_steps
        warmup_steps = int(effective_max_steps * 0.06)
        milestone1 = int(effective_max_steps * 0.8)
        milestone2 = int(effective_max_steps * 0.9)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            elif step < milestone1:
                return 1
            elif step < milestone2:
                return 0.316
            else:
                return 0.1

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    scaler = GradScaler() if config.use_amp else None

    # Initialize lists to store losses
    train_losses = []
    val_losses = []

    # Compute initial validation and train loss
    model.eval()
    initial_eval = evaluate_model(model, val_loader, config)
    val_losses.append(initial_eval['val_loss'])
    print(f"\nInitial Val Loss: {initial_eval['val_loss']:.4f}, "
          f"Val Acc: {initial_eval['val_accuracy']:.4f}, "
          f"Val PPL: {initial_eval['val_perplexity']:.2f}")

    # Compute initial train loss using evaluate_model
    initial_train_eval = evaluate_model(model, train_loader, config)
    train_losses.append(initial_train_eval['val_loss'])
    print(f"Initial Train Loss: {initial_train_eval['val_loss']:.4f}")

    # Training loop
    model.train()
    step = 0
    start_time = time.time()
    best_val_loss = float('inf')

    pbar = tqdm(total=config.max_steps, desc="Training")

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break
        
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = ((step + 1) % config.gradient_accumulation_steps == 0)
            if config.use_amp:
                with autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    loss = loss / config.gradient_accumulation_steps
                loss_detached = loss.detach().clone()
                if ddp:
                    with torch.no_grad():
                        dist.all_reduce(loss_detached, op=dist.ReduceOp.AVG)
                        
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / config.gradient_accumulation_steps
                
                loss_detached = loss.detach().clone()
                if ddp:
                    with torch.no_grad():
                        dist.all_reduce(loss_detached, op=dist.ReduceOp.AVG)

                loss.backward()

            # Optimizer step after accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    for optimizer in optimizers:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    for optimizer in optimizers:
                        scaler.step(optimizer)
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
            
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy_local = (predictions == y).float().mean().to(device)

                if ddp:
                    dist.reduce(accuracy_local, dst=0, op=dist.ReduceOp.SUM)

                current_loss_local = loss_detached.clone().to(device)
                if ddp:
                    dist.reduce(current_loss_local, dst=0, op=dist.ReduceOp.SUM)
                    
                    
                # Logging
                if is_master:
                    with torch.no_grad():
                        world_size = dist.get_world_size() if ddp else 1
                        accuracy = (accuracy_local / world_size).item()
                        current_loss = (current_loss_local / world_size).item() * config.gradient_accumulation_steps
                        perplexity = math.exp(min(current_loss, 20))

                    pbar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'acc': f'{accuracy:.3f}',
                        'ppl': f'{perplexity:.1f}',
                        'muon_lr': f'{optimizers[0].param_groups[0]["lr"]:.2e}',
                        'adamw_lr': f'{optimizers[1].param_groups[0]["lr"]:.2e}'
                    })

                # Evaluation
                if step % config.eval_every == 0 and step > 0:
                    model.eval()
                    eval_metrics = evaluate_model(model, val_loader, config)
                    val_losses.append(eval_metrics['val_loss'])
                    train_eval_metrics = evaluate_model(model, train_loader, config)
                    train_losses.append(train_eval_metrics['val_loss'])
                    print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                        f"Train Loss: {train_eval_metrics['val_loss']:.4f}, "
                        f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                        f"Val PPL: {eval_metrics['val_perplexity']:.2f}")
                    model.train()

                    if eval_metrics['val_loss'] < best_val_loss:
                        best_val_loss = eval_metrics['val_loss']

            step += 1
            if step % 50 == 0 and is_master:
                pbar.update(50)
    
    if ddp:
        dist.barrier()
        if not is_master:
            try:
                dist.destroy_process_group()
            except Exception:
                pass
            return

    # Final evaluation
    if is_master:
        pbar.close()
        training_time = time.time() - start_time
        print(f"   Training completed in {training_time:.1f} seconds")
        model.eval()
        final_eval = evaluate_model(model, val_loader, config)
        final_train_eval = evaluate_model(model, train_loader, config)
        val_losses.append(final_eval['val_loss'])
        train_losses.append(final_train_eval['val_loss'])
        print(f"   Final - Val Loss: {final_eval['val_loss']:.4f}, "
            f"Train Loss: {final_train_eval['val_loss']:.4f}, "
            f"Val Acc: {final_eval['val_accuracy']:.4f}, "
            f"Val PPL: {final_eval['val_perplexity']:.2f}")

        if ddp:
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        # Print stored losses
        print("\n Train Losses:", [f"{x:.4f}" for x in train_losses])
        print(" Validation Losses:", [f"{x:.4f}" for x in val_losses])

        return raw_model, final_eval