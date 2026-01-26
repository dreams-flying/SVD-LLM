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


def convert_blocks_to_trainable(model: nn.Module) -> int:
    """
    Convert block values from buffers to trainable parameters.
    Keep original values - don't normalize or clamp (would destroy residual compensation).

    Args:
        model: The compressed model

    Returns:
        Number of trainable block parameters
    """
    total_block_params = 0

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

                    # Keep original values and dtype - don't normalize or clamp!
                    # Blocks contain residual W_orig - W_svd, normalization destroys compensation
                    param = nn.Parameter(blocks_T.clone().detach())
                    module.register_parameter(buffer_name, param)

                    total_block_params += param.numel()

    return total_block_params


def freeze_non_block_params(model: nn.Module, train_layernorm: bool = True, train_bias: bool = True) -> None:
    """
    Freeze most parameters, but keep trainable:
    - Block values (g*_blocks_T)
    - LayerNorm parameters (helps adapt to block changes)
    - Bias terms (low cost, helps adaptation)

    Args:
        model: The model to freeze
        train_layernorm: Whether to train LayerNorm parameters
        train_bias: Whether to train bias parameters
    """
    for name, param in model.named_parameters():
        # Always train block values
        if 'blocks_T' in name:
            param.requires_grad = True
        # Train LayerNorm (input_layernorm, post_attention_layernorm, norm)
        elif train_layernorm and ('layernorm' in name.lower() or 'norm' in name.lower()):
            param.requires_grad = True
        # Train bias terms
        elif train_bias and 'bias' in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all trainable parameters (blocks, LayerNorm, bias).

    Args:
        model: The model

    Returns:
        List of trainable parameters
    """
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
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
    warmup_ratio: float = 0.1,
    gradient_accumulation: int = 4,
    max_grad_norm: float = 1.0,
    dataset_name: str = "wikitext2",
    train_layernorm: bool = True,
    train_bias: bool = True,
) -> nn.Module:
    """
    Fine-tune residual blocks (and optionally LayerNorm/bias).

    Args:
        model: Compressed model with SVDLinearWithDenseBlocks
        tokenizer: Tokenizer
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        seq_len: Sequence length
        num_samples: Number of calibration samples
        warmup_ratio: Warmup ratio (fraction of total steps)
        gradient_accumulation: Gradient accumulation steps
        max_grad_norm: Maximum gradient norm for clipping
        dataset_name: Dataset name
        train_layernorm: Whether to train LayerNorm parameters
        train_bias: Whether to train bias parameters

    Returns:
        Fine-tuned model
    """
    print("="*60)
    print("Block Fine-tuning")
    print("="*60)

    # Step 1: Convert blocks to trainable parameters (keep original values!)
    print("\nConverting blocks to trainable parameters...")
    num_block_params = convert_blocks_to_trainable(model)
    print(f"  Total block parameters: {num_block_params:,}")

    if num_block_params == 0:
        print("  No blocks found, skipping fine-tuning")
        return model

    # Step 2: Freeze non-block parameters (but keep LayerNorm/bias trainable)
    print("\nSetting up trainable parameters...")
    print(f"  Train LayerNorm: {train_layernorm}, Train bias: {train_bias}")
    freeze_non_block_params(model, train_layernorm=train_layernorm, train_bias=train_bias)

    # Count trainable params by type
    block_params_count = 0
    ln_params_count = 0
    bias_params_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'blocks_T' in name:
                block_params_count += param.numel()
            elif 'layernorm' in name.lower() or 'norm' in name.lower():
                ln_params_count += param.numel()
            elif 'bias' in name.lower():
                bias_params_count += param.numel()

    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Blocks: {block_params_count:,}, LayerNorm: {ln_params_count:,}, Bias: {bias_params_count:,}")
    print(f"  Trainable: {trainable_total:,} / {total_params:,} ({100*trainable_total/total_params:.2f}%)")

    # Step 3: Create data loader
    print(f"\nCreating calibration data ({dataset_name})...")
    dataloader = create_calib_dataloader(
        tokenizer, dataset_name, num_samples, seq_len, batch_size
    )
    print(f"  Samples: {len(dataloader.dataset)}, Batch size: {batch_size}")

    # Step 4: Setup optimizer
    trainable_params = get_trainable_params(model)
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01, eps=1e-8)

    # Learning rate scheduler with warmup (use ratio to avoid warmup > total_steps)
    total_steps = max(1, len(dataloader) * num_epochs // gradient_accumulation)
    warmup_steps = int(total_steps * warmup_ratio)
    print(f"  Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        # Linear decay after warmup
        decay_steps = total_steps - warmup_steps
        if decay_steps <= 0:
            return 1.0
        progress = (step - warmup_steps) / decay_steps
        return max(0.1, 1.0 - 0.9 * progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Step 5: Training loop
    # Don't call model.float() - keep original dtype to preserve compressed module behavior
    print(f"\nTraining for {num_epochs} epochs...")
    model.train()
    model = model.to(device)

    global_step = 0
    accumulated_loss = 0.0
    acc_steps = 0
    nan_count = 0
    max_nan_batches = 10

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                if outputs.loss is not None:
                    loss = outputs.loss
                else:
                    # Manual loss computation (no ignore_index since no padding in our data)
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
                if not check_gradients_valid(trainable_params):
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
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

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
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation=args.gradient_accumulation,
        dataset_name=args.dataset,
        train_layernorm=args.train_layernorm,
        train_bias=args.train_bias,
    )

    # Save model
    print("\n" + "="*60)
    print("Saving fine-tuned model...")

    # Ensure eval mode and move to CPU for saving
    model.eval()
    model = model.cpu()

    final_path = os.path.join(args.output_dir, "model_block_finetuned.pt")
    torch.save({'model': model, 'tokenizer': tokenizer}, final_path)
    print(f"  Saved to: {final_path}")

    # Evaluate if requested
    if args.evaluate:
        print("\nEvaluating fine-tuned model...")
        from evaluater import ppl_eval

        # Ensure model is in eval mode
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
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio (fraction of total steps)')
    parser.add_argument('--gradient_accumulation', type=int, default=4,
                        help='Gradient accumulation steps')

    # What to train
    parser.add_argument('--train_layernorm', action='store_true', default=True,
                        help='Train LayerNorm parameters (default: True)')
    parser.add_argument('--no_train_layernorm', action='store_false', dest='train_layernorm',
                        help='Do not train LayerNorm parameters')
    parser.add_argument('--train_bias', action='store_true', default=True,
                        help='Train bias parameters (default: True)')
    parser.add_argument('--no_train_bias', action='store_false', dest='train_bias',
                        help='Do not train bias parameters')

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
