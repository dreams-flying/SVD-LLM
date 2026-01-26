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
        --num_epochs 2 \
        --learning_rate 1e-4
"""

import os
import sys
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


def convert_blocks_to_trainable(model: nn.Module, max_block_norm: float = 10.0) -> int:
    """
    Convert block values from buffers to trainable parameters.
    Also normalizes large block values to prevent NaN during training.

    Args:
        model: The compressed model
        max_block_norm: Maximum norm for block values (clamp to prevent NaN)

    Returns:
        Number of trainable block parameters
    """
    total_block_params = 0
    normalized_count = 0

    for name, module in model.named_modules():
        if isinstance(module, SVDLinearWithDenseBlocks):
            if module.num_groups == 0:
                continue

            # Convert each block group's values to parameters
            for gi in range(module.num_groups):
                buffer_name = f"g{gi}_blocks_T"
                if hasattr(module, buffer_name):
                    # Get the buffer tensor
                    blocks_T = getattr(module, buffer_name)

                    # Delete the buffer
                    delattr(module, buffer_name)

                    # Clone and convert to FP32 for stable training
                    blocks_data = blocks_T.clone().detach().float()

                    # Normalize large block values to prevent NaN
                    # Block values come from W_orig - W_svd, can be very large
                    block_norm = blocks_data.norm()
                    if block_norm > max_block_norm:
                        scale_factor = max_block_norm / block_norm
                        blocks_data = blocks_data * scale_factor
                        normalized_count += 1

                    # Clamp extreme values
                    blocks_data = torch.clamp(blocks_data, -max_block_norm, max_block_norm)

                    # Register as parameter (trainable)
                    param = nn.Parameter(blocks_data)
                    module.register_parameter(buffer_name, param)

                    total_block_params += param.numel()

    if normalized_count > 0:
        print(f"  Normalized {normalized_count} blocks with large values")

    return total_block_params


def freeze_non_block_params(model: nn.Module) -> None:
    """
    Freeze all parameters except block values.

    Args:
        model: The model to freeze
    """
    for name, param in model.named_parameters():
        # Only train block values (g*_blocks_T)
        if 'blocks_T' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def get_trainable_block_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all trainable block parameters.

    Args:
        model: The model

    Returns:
        List of trainable parameters
    """
    params = []
    for name, param in model.named_parameters():
        if 'blocks_T' in name and param.requires_grad:
            params.append(param)
    return params


def create_calib_dataloader(tokenizer, dataset_name: str = "wikitext2",
                            num_samples: int = 256, seq_len: int = 512,
                            batch_size: int = 4) -> DataLoader:
    """
    Create calibration data loader with proper attention masks.

    Args:
        tokenizer: The tokenizer
        dataset_name: Dataset to use
        num_samples: Number of samples
        seq_len: Sequence length
        batch_size: Batch size

    Returns:
        DataLoader
    """
    # Ensure tokenizer has pad token
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

    # Tokenize
    tokens = tokenizer(text, return_tensors='pt').input_ids[0]

    # Create samples with attention masks
    samples = []
    for i in range(min(num_samples, len(tokens) // seq_len)):
        start = i * seq_len
        input_ids = tokens[start:start + seq_len]

        # Create attention mask (all 1s for non-padded sequences)
        attention_mask = torch.ones_like(input_ids)

        # Labels: use -100 for positions we don't want to compute loss on
        # For causal LM, we want to predict all tokens, so labels = input_ids
        labels = input_ids.clone()

        samples.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        })

    return DataLoader(samples, batch_size=batch_size, shuffle=True)


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
    num_epochs: int = 2,
    learning_rate: float = 1e-4,
    batch_size: int = 4,
    seq_len: int = 512,
    num_samples: int = 256,
    warmup_steps: int = 50,
    gradient_accumulation: int = 4,
    max_grad_norm: float = 1.0,
    dataset_name: str = "wikitext2",
) -> nn.Module:
    """
    Fine-tune residual blocks.

    Args:
        model: Compressed model with SVDLinearWithDenseBlocks
        tokenizer: Tokenizer
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        seq_len: Sequence length
        num_samples: Number of calibration samples
        warmup_steps: Warmup steps
        gradient_accumulation: Gradient accumulation steps
        max_grad_norm: Maximum gradient norm for clipping
        dataset_name: Dataset name

    Returns:
        Fine-tuned model
    """
    print("="*60)
    print("Block Fine-tuning")
    print("="*60)

    # Step 1: Convert model to FP32 for stable training
    print("\nConverting model to FP32 for stable training...")
    model = model.float()

    # Step 2: Convert blocks to trainable parameters (includes normalization)
    print("\nConverting blocks to trainable parameters...")
    num_block_params = convert_blocks_to_trainable(model, max_block_norm=10.0)
    print(f"  Total block parameters: {num_block_params:,}")

    if num_block_params == 0:
        print("  No blocks found, skipping fine-tuning")
        return model

    # Step 3: Freeze non-block parameters
    print("\nFreezing non-block parameters...")
    freeze_non_block_params(model)

    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Step 4: Create data loader
    print(f"\nCreating calibration data ({dataset_name})...")
    dataloader = create_calib_dataloader(
        tokenizer, dataset_name, num_samples, seq_len, batch_size
    )
    print(f"  Samples: {len(dataloader.dataset)}, Batch size: {batch_size}")

    # Step 5: Setup optimizer with eps for numerical stability
    block_params = get_trainable_block_params(model)
    optimizer = torch.optim.AdamW(block_params, lr=learning_rate, weight_decay=0.01, eps=1e-8)

    # Learning rate scheduler with warmup
    total_steps = len(dataloader) * num_epochs // gradient_accumulation
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Step 6: Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    model.train()
    model = model.to(device)

    global_step = 0
    accumulated_loss = 0.0
    acc_steps = 0
    nan_count = 0
    max_nan_batches = 10  # Stop if too many NaN batches

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass with attention_mask
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                if outputs.loss is not None:
                    loss = outputs.loss
                else:
                    # Manual loss computation with ignore_index for padding
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    # Use ignore_index=-100 for padding tokens
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
            except Exception as e:
                print(f"\n  Warning: Forward pass error: {e}")
                optimizer.zero_grad()
                continue

            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(f"\n  Warning: NaN/Inf loss detected (count: {nan_count})")
                optimizer.zero_grad()
                if nan_count >= max_nan_batches:
                    print(f"\n  Error: Too many NaN batches ({nan_count}), stopping training")
                    break
                continue

            loss = loss / gradient_accumulation
            loss.backward()

            accumulated_loss += loss.item()
            acc_steps += 1

            # Optimizer step
            if acc_steps >= gradient_accumulation:
                # Check for NaN/Inf gradients BEFORE clipping
                if not check_gradients_valid(block_params):
                    print(f"\n  Warning: NaN/Inf gradients detected, skipping update")
                    optimizer.zero_grad()
                    accumulated_loss = 0.0
                    acc_steps = 0
                    nan_count += 1
                    if nan_count >= max_nan_batches:
                        print(f"\n  Error: Too many NaN batches ({nan_count}), stopping training")
                        break
                    continue

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(block_params, max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                epoch_loss += accumulated_loss * gradient_accumulation
                num_batches += 1

                pbar.set_postfix({
                    'loss': f'{accumulated_loss * gradient_accumulation:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

                accumulated_loss = 0.0
                acc_steps = 0

        if nan_count >= max_nan_batches:
            break

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"  Epoch {epoch+1} average loss: {avg_loss:.4f}")

    # Set model to eval mode after training
    model.eval()

    print("\nBlock fine-tuning completed!")
    return model


def main(args):
    """Main function."""
    print("="*60)
    print("Block Fine-tuning for SVD-LLM")
    print("="*60)
    print(f"  Input model: {args.prune_model}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")

    # Load model
    print("\nLoading compressed model...")
    pruned_dict = torch.load(args.prune_model, map_location='cpu')
    tokenizer = pruned_dict['tokenizer']
    model = pruned_dict['model']

    # Ensure tokenizer has pad token
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

    # Create output directory
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
        warmup_steps=args.warmup_steps,
        gradient_accumulation=args.gradient_accumulation,
        dataset_name=args.dataset,
    )

    # Save model
    print("\n" + "="*60)
    print("Saving fine-tuned model...")

    # Ensure eval mode and move to CPU for saving
    model.eval()
    model = model.cpu().float()

    final_path = os.path.join(args.output_dir, "model_block_finetuned.pt")
    torch.save({'model': model, 'tokenizer': tokenizer}, final_path)
    print(f"  Saved to: {final_path}")

    # Evaluate if requested
    if args.evaluate:
        print("\nEvaluating fine-tuned model...")
        from evaluater import ppl_eval

        # Ensure model is in eval mode and FP32
        model.eval()
        model = model.float().to(device)

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
            print("  Model may need to be loaded fresh for evaluation")

    print("\n" + "="*60)
    print("Block fine-tuning completed!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Block Fine-tuning for SVD-LLM')

    # Model paths
    parser.add_argument('--prune_model', type=str, required=True,
                        help='Path to compressed model (.pt file)')
    parser.add_argument('--output_dir', type=str, default='./block_finetune_output',
                        help='Output directory')

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--seq_len', type=int, default=512,
                        help='Sequence length')
    parser.add_argument('--num_samples', type=int, default=256,
                        help='Number of calibration samples')
    parser.add_argument('--warmup_steps', type=int, default=50,
                        help='Warmup steps')
    parser.add_argument('--gradient_accumulation', type=int, default=4,
                        help='Gradient accumulation steps')

    # Data
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'c4'],
                        help='Training dataset')

    # Evaluation
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate after training')
    parser.add_argument('--model_seq_len', type=int, default=2048,
                        help='Model sequence length for evaluation')
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help='Evaluation batch size')

    args = parser.parse_args()
    main(args)
