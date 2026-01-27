#!/usr/bin/env python
# coding:utf8
"""
Block Fine-tuning for SVD-LLM

Fine-tune the residual blocks in SVDLinearWithDenseBlocks while keeping:
- SVD components (u_proj, v_proj) frozen
- Block positions fixed
- Only block VALUES trainable

This preserves the compression ratio while improving accuracy.

Structure: W ≈ U @ V + Σ blocks[i]
- U, V: frozen (from compression)
- block positions: fixed (from OMP selection)
- block values: TRAINABLE

Usage:
    python block_finetune.py \
        --prune_model path/to/compressed_model.pt \
        --output_dir ./block_finetune_output \
        --num_epochs 5 \
        --learning_rate 5e-4
"""

import os
import sys
import math
import argparse
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

from fisher_svd import SVDLinear, SVDLinearWithDenseBlocks

device = "cuda" if torch.cuda.is_available() else "cpu"


def convert_blocks_to_trainable(model: nn.Module) -> int:
    """
    Convert block values from buffers to trainable parameters.
    Only blocks are converted to FP32; model backbone stays BF16/FP16.
    """
    total_block_params = 0

    for name, module in model.named_modules():
        if isinstance(module, SVDLinearWithDenseBlocks):
            if module.num_groups == 0:
                continue

            for gi in range(module.num_groups):
                buffer_name = f"g{gi}_blocks_T"
                if hasattr(module, buffer_name):
                    blocks_T = getattr(module, buffer_name)
                    delattr(module, buffer_name)

                    # FP32 for stable gradients, keep original values
                    param = nn.Parameter(blocks_T.clone().detach().float())
                    module.register_parameter(buffer_name, param)

                    total_block_params += param.numel()

    return total_block_params


def freeze_non_block_params(model: nn.Module, train_layernorm: bool = True, train_bias: bool = True) -> None:
    """
    Freeze most parameters, unfreeze blocks + LayerNorm + bias.
    Trainable params converted to FP32 for stable gradients.
    """
    for name, param in model.named_parameters():
        if 'blocks_T' in name:
            param.requires_grad = True
        elif train_layernorm and ('layernorm' in name.lower() or 'norm' in name.lower()):
            param.requires_grad = True
            param.data = param.data.float()
        elif train_bias and 'bias' in name.lower():
            param.requires_grad = True
            param.data = param.data.float()
        else:
            param.requires_grad = False


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """Get all trainable parameters."""
    return [p for _, p in model.named_parameters() if p.requires_grad]


def create_calib_dataloader(tokenizer, dataset_name: str = "wikitext2",
                            num_samples: int = 256, seq_len: int = 512,
                            batch_size: int = 4) -> DataLoader:
    """Create calibration data loader with attention masks."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if dataset_name == "wikitext2":
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        text = "\n\n".join(data['text'])
    elif dataset_name == "c4":
        data = load_dataset('allenai/c4', 'en', split='train', streaming=True)
        texts = []
        for i, item in enumerate(data):
            if i >= num_samples:
                break
            texts.append(item['text'])
        text = "\n\n".join(texts)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    tokens = tokenizer(text, return_tensors='pt').input_ids[0]

    samples = []
    for i in range(min(num_samples, len(tokens) // seq_len)):
        start = i * seq_len
        input_ids = tokens[start:start + seq_len]
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        samples.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        })

    return DataLoader(samples, batch_size=batch_size, shuffle=True)


def verify_block_gradients(model: nn.Module) -> Dict[str, float]:
    """
    After first backward, verify blocks actually received gradients.
    Returns dict of {param_name: grad_norm}.
    """
    grad_info = {}
    has_grad = 0
    no_grad = 0
    for name, param in model.named_parameters():
        if 'blocks_T' in name and param.requires_grad:
            if param.grad is not None and param.grad.abs().sum() > 0:
                grad_info[name] = param.grad.norm().item()
                has_grad += 1
            else:
                grad_info[name] = 0.0
                no_grad += 1
    return grad_info, has_grad, no_grad


def check_gradients_valid(params) -> bool:
    """Check if gradients are valid (no NaN or Inf)."""
    for p in params:
        if p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                return False
    return True


def block_finetune(
    model: nn.Module,
    tokenizer,
    num_epochs: int = 5,
    learning_rate: float = 5e-4,
    batch_size: int = 2,
    seq_len: int = 512,
    num_samples: int = 256,
    warmup_ratio: float = 0.05,
    gradient_accumulation: int = 4,
    max_grad_norm: float = 1.0,
    dataset_name: str = "wikitext2",
    train_layernorm: bool = True,
    train_bias: bool = True,
    use_amp: bool = True,
) -> nn.Module:
    """
    Fine-tune residual blocks (and optionally LayerNorm/bias).

    NOTE: Gradient checkpointing is NOT used because
    SVDLinearWithDenseBlocks.forward uses in-place index_add_(),
    which is incompatible with gradient checkpointing (causes
    incorrect gradients when forward is re-executed during backward).

    Memory is saved by: AMP + small batch_size + gradient_accumulation.
    """
    print("=" * 60)
    print("Block Fine-tuning (AMP)" if use_amp else "Block Fine-tuning")
    print("=" * 60)

    # ---- WARNING: DO NOT enable gradient_checkpointing ----
    # SVDLinearWithDenseBlocks.forward uses y2d.index_add_() (in-place op).
    # gradient_checkpointing re-runs forward during backward,
    # which corrupts gradients on in-place modified tensors.
    # -------------------------------------------------------

    # Step 1: Convert blocks buffer -> trainable parameter (FP32)
    print("\nStep 1: Converting blocks to trainable parameters...")
    num_block_params = convert_blocks_to_trainable(model)
    print(f"  Total block parameters: {num_block_params:,}")

    if num_block_params == 0:
        print("  No blocks found, skipping fine-tuning")
        return model

    # Step 2: Freeze non-block parameters
    print("\nStep 2: Setting up trainable parameters...")
    print(f"  Train LayerNorm: {train_layernorm}, Train bias: {train_bias}")
    freeze_non_block_params(model, train_layernorm=train_layernorm, train_bias=train_bias)

    # Count trainable params by type
    block_cnt = 0
    ln_cnt = 0
    bias_cnt = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'blocks_T' in name:
                block_cnt += param.numel()
            elif 'layernorm' in name.lower() or 'norm' in name.lower():
                ln_cnt += param.numel()
            elif 'bias' in name.lower():
                bias_cnt += param.numel()

    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Blocks: {block_cnt:,}, LayerNorm: {ln_cnt:,}, Bias: {bias_cnt:,}")
    print(f"  Trainable: {trainable_total:,} / {total_params:,} ({100 * trainable_total / total_params:.2f}%)")

    # Step 3: Create data loader
    print(f"\nStep 3: Creating calibration data ({dataset_name})...")
    dataloader = create_calib_dataloader(
        tokenizer, dataset_name, num_samples, seq_len, batch_size
    )
    print(f"  Samples: {len(dataloader.dataset)}, Batch size: {batch_size}")

    # Step 4: Setup optimizer
    trainable_params = get_trainable_params(model)
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01, eps=1e-8)

    # Cosine annealing scheduler with warmup
    total_steps = max(1, len(dataloader) * num_epochs // gradient_accumulation)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    print(f"  Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Step 5: Setup AMP
    use_cuda = device == "cuda" and torch.cuda.is_available()
    if use_amp and use_cuda:
        # Detect backbone dtype (skip FP32 trainable params)
        model_dtype = torch.float32
        for p in model.parameters():
            if p.dtype in (torch.float16, torch.bfloat16):
                model_dtype = p.dtype
                break
        if model_dtype == torch.bfloat16:
            amp_dtype = torch.bfloat16
            scaler = None  # BF16 doesn't need GradScaler
        else:
            amp_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler()
        print(f"  AMP: autocast={amp_dtype}, GradScaler={scaler is not None}")
    else:
        amp_dtype = torch.float32
        scaler = None
        if use_amp:
            print("  AMP disabled (no CUDA)")

    # Step 6: Training
    print(f"\nStep 6: Training for {num_epochs} epochs...")
    model.train()
    model = model.to(device)

    # Evaluate initial loss
    print("\n  Evaluating initial loss...")
    model.eval()
    with torch.no_grad():
        init_losses = []
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labs = batch['labels'].to(device)
            with torch.cuda.amp.autocast(enabled=use_amp and use_cuda, dtype=amp_dtype):
                out = model(input_ids=ids, attention_mask=mask, labels=labs, use_cache=False)
            if out.loss is not None:
                init_losses.append(out.loss.float().item())
        if init_losses:
            print(f"  Initial loss: {sum(init_losses) / len(init_losses):.4f}")
    model.train()

    global_step = 0
    accumulated_loss = 0.0
    acc_steps = 0
    nan_count = 0
    max_nan_batches = 10
    best_loss = float('inf')
    grad_verified = False

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward with AMP autocast, use_cache=False to save memory
            try:
                with torch.cuda.amp.autocast(enabled=use_amp and use_cuda, dtype=amp_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                    )

                    if outputs.loss is not None:
                        loss = outputs.loss
                    else:
                        logits = outputs.logits
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
            except Exception as e:
                print(f"\n  Warning: Forward pass error: {e}")
                optimizer.zero_grad()
                if scaler:
                    scaler.update()
                continue

            # NaN/Inf check
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(f"\n  Warning: NaN/Inf loss (count: {nan_count})")
                optimizer.zero_grad()
                if scaler:
                    scaler.update()
                if nan_count >= max_nan_batches:
                    print(f"\n  Error: Too many NaN batches, stopping")
                    break
                continue

            loss = loss / gradient_accumulation

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.float().item()
            acc_steps += 1

            # Verify gradients on first backward
            if not grad_verified:
                grad_info, has_grad, no_grad = verify_block_gradients(model)
                if has_grad > 0:
                    sample_grads = list(grad_info.items())[:3]
                    grad_strs = [f"{n.split('.')[-3]}.{n.split('.')[-1]}={v:.2e}" for n, v in sample_grads]
                    print(f"\n  Gradient check: {has_grad} blocks have grad, {no_grad} blocks have no grad")
                    print(f"  Sample grad norms: {', '.join(grad_strs)}")
                else:
                    print(f"\n  WARNING: No blocks received gradients! Training will have no effect.")
                grad_verified = True

            # Optimizer step
            if acc_steps >= gradient_accumulation:
                if scaler:
                    scaler.unscale_(optimizer)

                if not check_gradients_valid(trainable_params):
                    print(f"\n  Warning: NaN/Inf gradients, skipping update")
                    optimizer.zero_grad()
                    if scaler:
                        scaler.update()
                    accumulated_loss = 0.0
                    acc_steps = 0
                    nan_count += 1
                    if nan_count >= max_nan_batches:
                        break
                    continue

                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                avg_batch_loss = accumulated_loss
                epoch_loss += avg_batch_loss
                num_batches += 1

                pbar.set_postfix({
                    'loss': f'{avg_batch_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

                accumulated_loss = 0.0
                acc_steps = 0

            # Free memory
            del outputs, loss
            if use_cuda:
                torch.cuda.empty_cache()

        if nan_count >= max_nan_batches:
            break

        avg_loss = epoch_loss / max(num_batches, 1)
        mark = "improved" if avg_loss < best_loss else "no improvement"
        if avg_loss < best_loss:
            best_loss = avg_loss
        print(f"  Epoch {epoch + 1} avg loss: {avg_loss:.4f} ({mark}, best: {best_loss:.4f})")

    model.eval()
    if use_cuda:
        torch.cuda.empty_cache()

    print("\nBlock fine-tuning completed!")
    return model


def main(args):
    """Main function."""
    print("=" * 60)
    print("Block Fine-tuning for SVD-LLM")
    print("=" * 60)
    print(f"  Input model: {args.prune_model}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")

    # Load model
    print("\nLoading compressed model...")
    pruned_dict = torch.load(args.prune_model, map_location='cpu')
    tokenizer = pruned_dict['tokenizer']
    model = pruned_dict['model']

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Count blocks
    num_blocks = 0
    for name, module in model.named_modules():
        if isinstance(module, SVDLinearWithDenseBlocks):
            num_blocks += module.num_groups
    print(f"  Found {num_blocks} block groups in model")

    if num_blocks == 0:
        print("  No blocks found, nothing to fine-tune")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Fine-tune
    model = block_finetune(
        model=model,
        tokenizer=tokenizer,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_samples=args.num_samples,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation=args.gradient_accumulation,
        dataset_name=args.dataset,
        train_layernorm=args.train_layernorm,
        train_bias=args.train_bias,
        use_amp=args.use_amp,
    )

    # Save model
    print("\n" + "=" * 60)
    print("Saving fine-tuned model...")

    model.eval()
    model = model.cpu()

    # Convert FP32 trainable params back to original dtype (BF16/FP16) for saving
    # Training used FP32 for gradient stability, but inference only needs BF16/FP16
    save_dtype = torch.bfloat16  # default
    for p in model.parameters():
        if p.dtype in (torch.float16, torch.bfloat16):
            save_dtype = p.dtype
            break
    fp32_converted = 0
    for name, param in model.named_parameters():
        if param.dtype == torch.float32 and ('blocks_T' in name
                or 'layernorm' in name.lower() or 'norm' in name.lower()
                or 'bias' in name.lower()):
            param.data = param.data.to(save_dtype)
            fp32_converted += 1
    print(f"  Converted {fp32_converted} FP32 params back to {save_dtype} for saving")

    final_path = os.path.join(args.output_dir, "model_block_finetuned.pt")
    torch.save({'model': model, 'tokenizer': tokenizer}, final_path)
    print(f"  Saved to: {final_path}")

    # Evaluate
    if args.evaluate:
        print("\nEvaluating fine-tuned model...")
        from evaluater import ppl_eval

        model.eval()
        model = model.to(device)

        try:
            ppl_eval(
                model, tokenizer,
                datasets=['wikitext2'],
                model_seq_len=args.model_seq_len,
                batch_size=args.eval_batch_size,
                device=device
            )
        except Exception as e:
            print(f"  Evaluation error: {e}")

    print("\n" + "=" * 60)
    print("Block fine-tuning completed!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Block Fine-tuning for SVD-LLM')

    parser.add_argument('--prune_model', type=str, required=True,
                        help='Path to compressed model (.pt file)')
    parser.add_argument('--output_dir', type=str, default='./block_finetune_output',
                        help='Output directory')

    # Training
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs (default: 5)')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Learning rate (default: 5e-4)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (default: 2, keep small for memory)')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Sequence length')
    parser.add_argument('--num_samples', type=int, default=256,
                        help='Number of calibration samples')
    parser.add_argument('--warmup_ratio', type=float, default=0.05,
                        help='Warmup ratio (default: 0.05)')
    parser.add_argument('--gradient_accumulation', type=int, default=4,
                        help='Gradient accumulation steps (default: 4)')

    # What to train
    parser.add_argument('--train_layernorm', action='store_true', default=True,
                        help='Train LayerNorm parameters (default: True)')
    parser.add_argument('--no_train_layernorm', action='store_false', dest='train_layernorm')
    parser.add_argument('--train_bias', action='store_true', default=True,
                        help='Train bias parameters (default: True)')
    parser.add_argument('--no_train_bias', action='store_false', dest='train_bias')

    # AMP
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision (default: True)')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp')

    # Data
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'c4'])

    # Evaluation
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--model_seq_len', type=int, default=2048)
    parser.add_argument('--eval_batch_size', type=int, default=1)

    args = parser.parse_args()
    main(args)
