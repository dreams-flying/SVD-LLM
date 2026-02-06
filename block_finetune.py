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
import random
import argparse
from typing import List, Dict, Tuple

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


# ---------------------------------------------------------------------------
# Custom LoRA implementation (works with both nn.Linear and SVDLinear*)
# ---------------------------------------------------------------------------

class LoRALayer(nn.Module):
    """
    LoRA adapter that wraps any module and adds a parallel low-rank path.

    Output = wrapped(x) + lora_B(lora_A(dropout(x))) * scaling

    Works with nn.Linear, SVDLinear, SVDLinearWithDenseBlocks, etc.
    """

    def __init__(
        self,
        wrapped: nn.Module,
        in_features: int,
        out_features: int,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.wrapped = wrapped
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # LoRA matrices (FP32 for stable gradients)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize: A with kaiming, B with zeros (initial output = original)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = self.wrapped(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return original + lora_out


def merge_lora_weights(model: nn.Module) -> int:
    """
    Merge LoRA adapters back into wrapped modules and remove LoRA wrappers.

    For nn.Linear: W_new = W_old + scaling * lora_B.weight @ lora_A.weight
    For SVDLinearWithDenseBlocks: expand SVD rank by r to absorb LoRA.

    Returns number of merged LoRA layers.
    """
    merged = 0

    for parent_name, parent_module in model.named_modules():
        for attr_name, child_module in list(parent_module.named_children()):
            if not isinstance(child_module, LoRALayer):
                continue

            lora = child_module
            wrapped = lora.wrapped
            scaling = lora.scaling

            # Compute LoRA delta: [out_features, in_features]
            # lora_B.weight: [out_features, r], lora_A.weight: [r, in_features]
            with torch.no_grad():
                delta = (lora.lora_B.weight @ lora.lora_A.weight) * scaling

            if isinstance(wrapped, nn.Linear):
                # Simple merge: add delta to weight
                wrapped.weight.data.add_(delta.to(wrapped.weight.dtype))
                setattr(parent_module, attr_name, wrapped)
                merged += 1

            elif isinstance(wrapped, (SVDLinear, SVDLinearWithDenseBlocks)):
                # Expand SVD rank to absorb LoRA:
                # Original: y = u_proj(v_proj(x)) [+ blocks]
                # LoRA:     y += scaling * lora_B @ lora_A @ x
                # Merged:   y = [u_proj | scaling*lora_B] @ [v_proj; lora_A] @ x [+ blocks]
                old_v = wrapped.v_proj
                old_u = wrapped.u_proj
                dtype = old_v.weight.dtype
                device = old_v.weight.device
                r = lora.r

                # New v_proj weight: [rank+r, in_features]
                new_v_weight = torch.cat([
                    old_v.weight.data,
                    lora.lora_A.weight.data.to(dtype=dtype, device=device),
                ], dim=0)

                # New u_proj weight: [out_features, rank+r]
                new_u_weight = torch.cat([
                    old_u.weight.data,
                    (lora.lora_B.weight.data * scaling).to(dtype=dtype, device=device),
                ], dim=1)

                # Directly replace weight data instead of creating new Linear
                # This preserves the original device and avoids FP32 initialization
                old_v.weight = nn.Parameter(new_v_weight)
                old_u.weight = nn.Parameter(new_u_weight)

                # Update in_features/out_features for the Linear modules
                old_v.out_features = new_v_weight.shape[0]
                old_u.in_features = new_u_weight.shape[1]

                # Handle bias if exists
                if old_v.bias is not None:
                    new_v_bias = torch.cat([
                        old_v.bias.data,
                        torch.zeros(r, dtype=dtype, device=device),
                    ])
                    old_v.bias = nn.Parameter(new_v_bias)

                setattr(parent_module, attr_name, wrapped)
                merged += 1

            full_name = f"{parent_name}.{attr_name}" if parent_name else attr_name
            print(f"    Merged LoRA -> {full_name}")

    return merged


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


def freeze_non_block_params(
    model: nn.Module,
    train_layernorm: bool = True,
    train_bias: bool = True,
) -> None:
    """
    Freeze most parameters, selectively unfreeze components.

    Args:
        model: The model to modify
        train_layernorm: Unfreeze LayerNorm parameters
        train_bias: Unfreeze bias parameters

    Note: LoRA parameters (if any) are handled by LoRALayer wrapper.
    Trainable params converted to FP32 for stable gradients.
    """
    for name, param in model.named_parameters():
        if 'blocks_T' in name:
            # Always train blocks
            param.requires_grad = True
        elif 'lora_' in name:
            # LoRA parameters are always trained
            param.requires_grad = True
            # Keep LoRA params in their original dtype
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


def load_and_tokenize_dataset(
    tokenizer,
    dataset_name: str = "wikitext2",
    split: str = 'train',
    max_docs: int = 10000,
) -> torch.Tensor:
    """
    Load dataset and return tokenized tensor.
    Separate from sampling to allow epoch-wise resampling.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset with proper split handling
    if dataset_name == "wikitext2":
        # wikitext2 splits: 'train', 'validation', 'test'
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        text = "\n\n".join(data['text'])
        print(f"    Loaded wikitext2 {split}: {len(data)} docs, {len(text):,} chars")
    elif dataset_name == "c4":
        # C4 splits: 'train', 'validation'
        c4_split = 'validation' if split in ('validation', 'val', 'test') else 'train'
        data = load_dataset('allenai/c4', 'en', split=c4_split, streaming=True)
        texts = []
        for i, item in enumerate(data):
            if i >= max_docs:
                break
            texts.append(item['text'])
        text = "\n\n".join(texts)
        print(f"    Loaded c4 {c4_split}: {len(texts)} docs, {len(text):,} chars")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Tokenize (no truncation - use full text)
    tokens = tokenizer(text, return_tensors='pt', truncation=False, add_special_tokens=False).input_ids[0]
    print(f"    Tokenized {split}: {len(tokens):,} tokens")
    return tokens


def sample_from_tokens(
    tokens: torch.Tensor,
    num_samples: int,
    seq_len: int,
    seed: int,
    is_train: bool = True,
) -> List[Dict[str, torch.Tensor]]:
    """
    Sample sequences from tokenized text.
    Can be called multiple times with different seeds for epoch-wise resampling.
    """
    total_tokens = len(tokens)
    if total_tokens < seq_len:
        raise ValueError(f"Dataset too small: {total_tokens} tokens < seq_len {seq_len}")

    max_start = total_tokens - seq_len
    max_non_overlap = max_start // seq_len + 1
    rng = random.Random(seed)

    if is_train:
        # Training: random sampling allows overlap for more diversity
        start_positions = [rng.randint(0, max_start) for _ in range(num_samples)]
    else:
        # Validation: evenly spaced, non-overlapping if possible
        if num_samples <= max_non_overlap:
            step = max_start // num_samples if num_samples > 1 else 0
            start_positions = [i * step for i in range(num_samples)]
        else:
            step = seq_len
            start_positions = [i * step for i in range(max_non_overlap)]

    samples = []
    for start in start_positions:
        input_ids = tokens[start:start + seq_len]
        if len(input_ids) < seq_len:
            continue
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        samples.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        })

    return samples


def create_dataloader(
    tokenizer,
    dataset_name: str = "wikitext2",
    num_samples: int = 256,
    seq_len: int = 512,
    batch_size: int = 4,
    split: str = 'train',
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create data loader with random sampling for better coverage.
    For simple usage - loads, tokenizes, samples, and creates DataLoader.
    """
    tokens = load_and_tokenize_dataset(tokenizer, dataset_name, split)
    is_train = (split == 'train')
    samples = sample_from_tokens(tokens, num_samples, seq_len, seed, is_train)

    if len(samples) < num_samples:
        max_non_overlap = (len(tokens) - seq_len) // seq_len + 1
        if is_train:
            print(f"    Warning: Only created {len(samples)}/{num_samples} samples")
        else:
            print(f"    Note: Created {len(samples)} val samples (max non-overlapping: {max_non_overlap})")

    print(f"    Created {len(samples)} samples from {len(tokens):,} tokens ({split})")
    return DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )


@torch.no_grad()
def validate(model, dataloader, use_amp: bool, amp_dtype, device: str) -> float:
    """
    Run validation and return average loss.
    Uses same loss calculation as training (manual cross_entropy) for consistency,
    but WITHOUT label_smoothing to get true loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    use_cuda = device == "cuda" and torch.cuda.is_available()

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp and use_cuda, dtype=amp_dtype):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,  # Compute loss manually for consistency with training
                use_cache=False,
            )
            # Same calculation as training, but no label_smoothing
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        total_loss += loss.float().item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def verify_block_gradients(model: nn.Module) -> Tuple[Dict[str, float], int, int]:
    """
    After first backward, verify blocks actually received gradients.
    Returns (grad_info, has_grad_count, no_grad_count).
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


def check_gradients_valid(params: List[nn.Parameter]) -> bool:
    """Check if gradients are valid (no NaN or Inf)."""
    for p in params:
        if p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                return False
    return True


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float = 0.1):
    """
    Create a schedule with linear warmup and cosine decay.

    Args:
        min_lr_ratio: Minimum LR as ratio of initial LR (default 0.1 = 10% of initial LR)
                      This prevents LR from decaying to 0.
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step + 1) / float(max(1, num_warmup_steps))
        # Cosine decay phase with minimum LR
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(1.0, progress)  # Clamp to [0, 1]
        # Decay from 1.0 to min_lr_ratio (not to 0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps: int):
    """Constant LR after warmup - often works better for fine-tuning."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step + 1) / float(max(1, num_warmup_steps))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def apply_lora(
    model: nn.Module,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
) -> nn.Module:
    """
    Apply LoRA adapters to target modules in the model.

    Supports both nn.Linear and SVDLinearWithDenseBlocks (custom SVD modules).
    For each target module name, finds matching modules and wraps them with LoRALayer.

    Args:
        model: Base model
        lora_r: LoRA rank (higher = more capacity, more params)
        lora_alpha: LoRA scaling factor (alpha/r is the scaling)
        lora_dropout: Dropout for LoRA layers
        lora_target_modules: Module names to apply LoRA (default: ["q_proj", "v_proj"])

    Returns:
        Model with LoRA adapters applied (in-place modification)
    """
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj"]

    target_set = set(lora_target_modules)

    print(f"\n  Applying LoRA adapters...")
    print(f"    r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"    target_modules={lora_target_modules}")

    lora_count = 0
    lora_total_params = 0

    # Walk the module tree and wrap matching modules
    # IMPORTANT: Skip children of SVD modules and LoRALayer to avoid name collision.
    # SVDLinearWithDenseBlocks has internal v_proj/u_proj which are SVD components,
    # NOT the attention projections we want to apply LoRA to.
    skip_types = (SVDLinear, SVDLinearWithDenseBlocks, LoRALayer)

    for parent_name, parent_module in model.named_modules():
        # Don't apply LoRA inside SVD modules or existing LoRA wrappers
        if isinstance(parent_module, skip_types):
            continue

        for attr_name, child_module in list(parent_module.named_children()):
            if attr_name not in target_set:
                continue

            # Determine in/out features
            if isinstance(child_module, nn.Linear):
                in_f = child_module.in_features
                out_f = child_module.out_features
            elif isinstance(child_module, (SVDLinear, SVDLinearWithDenseBlocks)):
                in_f = child_module.v_proj.in_features
                out_f = child_module.u_proj.out_features
            else:
                continue

            # Wrap with LoRA
            lora_module = LoRALayer(
                wrapped=child_module,
                in_features=in_f,
                out_features=out_f,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            setattr(parent_module, attr_name, lora_module)

            params = lora_r * (in_f + out_f)
            lora_total_params += params
            lora_count += 1

            full_name = f"{parent_name}.{attr_name}" if parent_name else attr_name
            print(f"    LoRA -> {full_name} ({type(child_module).__name__}, "
                  f"in={in_f}, out={out_f}, +{params:,} params)")

    if lora_count == 0:
        print(f"    WARNING: No modules matched target_modules={lora_target_modules}")
        print(f"    Available top-level modules (excluding SVD internals):")
        for n, m in model.named_modules():
            if isinstance(m, skip_types):
                continue
            for cn, cm in m.named_children():
                if isinstance(cm, (nn.Linear, SVDLinear, SVDLinearWithDenseBlocks)):
                    full = f"{n}.{cn}" if n else cn
                    print(f"      {full} ({type(cm).__name__})")
    else:
        print(f"    Total: {lora_count} LoRA adapters, {lora_total_params:,} parameters")

    return model


def block_finetune(
    model: nn.Module,
    tokenizer,
    num_epochs: int = 5,
    learning_rate: float = 5e-4,
    block_lr_multiplier: float = 1.0,
    lora_lr_multiplier: float = 1.0,
    batch_size: int = 2,
    seq_len: int = 512,
    num_samples: int = 256,
    warmup_ratio: float = 0.05,
    gradient_accumulation: int = 4,
    max_grad_norm: float = 1.0,
    weight_decay: float = 0.0,
    block_weight_decay: float = 0.0,
    ln_weight_decay: float = 0.0,
    lora_weight_decay: float = 0.0,
    dataset_name: str = "wikitext2",
    train_layernorm: bool = True,
    train_bias: bool = True,
    # LoRA parameters
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    use_amp: bool = True,
    scheduler_type: str = "cosine",
    min_lr_ratio: float = 0.1,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """
    Fine-tune residual blocks and optionally LoRA adapters.

    Key parameters for better training:
    - block_lr_multiplier: Scale blocks LR relative to base LR (default 1.0)
    - lora_lr_multiplier: Scale LoRA LR relative to base LR (default 1.0)
    - block_weight_decay: Weight decay for blocks (default 0.0 - blocks shouldn't be regularized)
    - lora_weight_decay: Weight decay for LoRA params (default 0.0)
    - scheduler_type: "cosine" or "constant" (constant often works better for fine-tuning)
    - min_lr_ratio: For cosine, minimum LR as ratio of initial (default 0.1 = don't decay to 0)
    - label_smoothing: Label smoothing for cross-entropy (default 0.0, try 0.1 for better generalization)

    LoRA parameters (use_lora=True to enable):
    - lora_r: LoRA rank (default 16, higher = more capacity)
    - lora_alpha: LoRA scaling factor (default 32)
    - lora_dropout: LoRA dropout (default 0.05)
    - lora_target_modules: Modules to apply LoRA (default: ["q_proj", "v_proj"])

    Memory is saved by: AMP + small batch_size + gradient_accumulation.
    """
    print("=" * 60)
    print("Block Fine-tuning (AMP)" if use_amp else "Block Fine-tuning")
    print("=" * 60)

    # Step 1: Convert blocks buffer -> trainable parameter (FP32)
    print("\nStep 1: Converting blocks to trainable parameters...")
    num_block_params = convert_blocks_to_trainable(model)
    print(f"  Total block parameters: {num_block_params:,}")

    if num_block_params == 0 and not use_lora:
        print("  No blocks found and LoRA disabled, skipping fine-tuning")
        return model

    # Step 1.5: Apply LoRA adapters (before freezing)
    if use_lora:
        model = apply_lora(
            model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )

    # Step 2: Freeze non-block parameters
    print("\nStep 2: Setting up trainable parameters...")
    print(f"  Train LayerNorm: {train_layernorm}, Train bias: {train_bias}")
    print(f"  Use LoRA: {use_lora}")

    freeze_non_block_params(
        model,
        train_layernorm=train_layernorm,
        train_bias=train_bias,
    )

    # Count trainable params by type
    block_cnt = 0
    ln_cnt = 0
    bias_cnt = 0
    lora_cnt = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'blocks_T' in name:
                block_cnt += param.numel()
            elif 'lora_' in name:
                lora_cnt += param.numel()
            elif 'layernorm' in name.lower() or 'norm' in name.lower():
                ln_cnt += param.numel()
            elif 'bias' in name.lower():
                bias_cnt += param.numel()

    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Blocks: {block_cnt:,}, LayerNorm: {ln_cnt:,}, Bias: {bias_cnt:,}")
    if lora_cnt > 0:
        print(f"  LoRA: {lora_cnt:,}")
    print(f"  Trainable: {trainable_total:,} / {total_params:,} ({100 * trainable_total / total_params:.2f}%)")

    # Step 3: Load and tokenize datasets (once), sampling done per-epoch
    print(f"\nStep 3: Loading datasets ({dataset_name})...")
    train_tokens = load_and_tokenize_dataset(tokenizer, dataset_name, split='train')
    val_tokens = load_and_tokenize_dataset(tokenizer, dataset_name, split='validation')
    print(f"  Train tokens: {len(train_tokens):,}, Val tokens: {len(val_tokens):,}")

    # Create validation loader (fixed samples for consistent evaluation)
    val_samples = sample_from_tokens(val_tokens, num_samples=128, seq_len=seq_len, seed=42, is_train=False)
    val_loader = DataLoader(
        val_samples, batch_size=batch_size, shuffle=False,
        pin_memory=torch.cuda.is_available(), num_workers=0,
    )
    print(f"  Val samples: {len(val_samples)} (fixed for consistent evaluation)")

    # Step 4: Setup optimizer with separate param groups
    # - Blocks: compensate SVD error, no weight decay
    # - LoRA: low-rank adapters, same LR or slightly different
    # - LayerNorm: scale/shift, typically no weight decay
    # - Bias: typically no weight decay
    block_params = []
    lora_params = []
    ln_params = []
    bias_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'blocks_T' in name:
                block_params.append(param)
            elif 'lora_' in name:
                lora_params.append(param)
            elif 'layernorm' in name.lower() or 'norm' in name.lower():
                ln_params.append(param)
            else:
                bias_params.append(param)

    param_groups = []
    if block_params:
        param_groups.append({
            'params': block_params,
            'lr': learning_rate * block_lr_multiplier,
            'weight_decay': block_weight_decay,
            'name': 'blocks'
        })
    if lora_params:
        param_groups.append({
            'params': lora_params,
            'lr': learning_rate * lora_lr_multiplier,
            'weight_decay': lora_weight_decay,
            'name': 'lora'
        })
    if ln_params:
        param_groups.append({
            'params': ln_params,
            'lr': learning_rate,
            'weight_decay': ln_weight_decay,
            'name': 'layernorm'
        })
    if bias_params:
        param_groups.append({
            'params': bias_params,
            'lr': learning_rate,
            'weight_decay': weight_decay,
            'name': 'bias'
        })

    optimizer = torch.optim.AdamW(param_groups, eps=1e-8)
    trainable_params = block_params + lora_params + ln_params + bias_params
    print(f"\nStep 4: Optimizer setup...")
    print(f"  Block params: {len(block_params)}, LR={learning_rate * block_lr_multiplier:.2e}, WD={block_weight_decay}")
    if lora_params:
        print(f"  LoRA params: {len(lora_params)}, LR={learning_rate * lora_lr_multiplier:.2e}, WD={lora_weight_decay}")
    print(f"  LayerNorm params: {len(ln_params)}, LR={learning_rate:.2e}, WD={ln_weight_decay}")
    print(f"  Bias params: {len(bias_params)}, LR={learning_rate:.2e}, WD={weight_decay}")

    # Calculate total steps based on num_samples
    batches_per_epoch = (num_samples + batch_size - 1) // batch_size
    total_steps = batches_per_epoch * num_epochs // gradient_accumulation
    warmup_steps = int(total_steps * warmup_ratio)
    print(f"  Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    print(f"  Scheduler: {scheduler_type}, min_lr_ratio: {min_lr_ratio}")

    if scheduler_type == "constant":
        scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio)

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

    # Initial validation
    print("\n  Initial validation...")
    init_val_loss = validate(model, val_loader, use_amp, amp_dtype, device)
    print(f"  Initial val_loss: {init_val_loss:.4f}")
    model.train()

    global_step = 0
    acc_steps = 0
    nan_count = 0
    max_nan_batches = 10
    best_val_loss = float('inf')
    grad_verified = False

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Resample training data each epoch for better coverage
        epoch_seed = 42 + epoch  # Different seed each epoch
        train_samples = sample_from_tokens(train_tokens, num_samples, seq_len, epoch_seed, is_train=True)
        train_loader = DataLoader(
            train_samples, batch_size=batch_size, shuffle=True,
            pin_memory=use_cuda, num_workers=0,
        )

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward with AMP autocast
            try:
                with torch.cuda.amp.autocast(enabled=use_amp and use_cuda, dtype=amp_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None,  # Compute loss ourselves for label smoothing
                        use_cache=False,
                    )

                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    # Loss for backward (with label_smoothing for regularization)
                    loss_for_backward = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        label_smoothing=label_smoothing,
                    )

                    # Loss for logging (without label_smoothing, consistent with val_loss)
                    with torch.no_grad():
                        loss_for_logging = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                        )
            except Exception as e:
                print(f"\n  Warning: Forward pass error: {e}")
                optimizer.zero_grad()
                if scaler:
                    scaler.update()
                continue

            # NaN/Inf check
            if torch.isnan(loss_for_backward) or torch.isinf(loss_for_backward):
                nan_count += 1
                print(f"\n  Warning: NaN/Inf loss (count: {nan_count})")
                optimizer.zero_grad()
                if scaler:
                    scaler.update()
                if nan_count >= max_nan_batches:
                    print(f"\n  Error: Too many NaN batches, stopping")
                    break
                continue

            # Scale loss for gradient accumulation
            scaled_loss = loss_for_backward / gradient_accumulation

            if scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # Accumulate unsmoothed loss for logging (same as val_loss calculation)
            epoch_loss += loss_for_logging.float().item()
            num_batches += 1
            acc_steps += 1

            # Verify gradients on first backward
            if not grad_verified:
                grad_info, has_grad, no_grad = verify_block_gradients(model)
                if has_grad > 0:
                    sample_grads = list(grad_info.items())[:3]
                    grad_strs = [f"{n.split('.')[-3]}.{n.split('.')[-1]}={v:.2e}" for n, v in sample_grads]
                    print(f"\n  Gradient check: {has_grad} blocks have grad, {no_grad} blocks no grad")
                    print(f"  Sample grad norms: {', '.join(grad_strs)}")
                else:
                    print(f"\n  WARNING: No blocks received gradients!")
                grad_verified = True

            # Update progress bar with current batch loss
            pbar.set_postfix({
                'loss': f'{loss_for_logging.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

            # Optimizer step
            if acc_steps >= gradient_accumulation:
                if scaler:
                    scaler.unscale_(optimizer)

                if not check_gradients_valid(trainable_params):
                    print(f"\n  Warning: NaN/Inf gradients, skipping update")
                    optimizer.zero_grad()
                    if scaler:
                        scaler.update()
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
                acc_steps = 0

            # Clean up
            del outputs, loss_for_backward, loss_for_logging

        if nan_count >= max_nan_batches:
            break

        # Epoch end: compute stats and validate
        avg_train_loss = epoch_loss / max(num_batches, 1)

        # Validation
        val_loss = validate(model, val_loader, use_amp, amp_dtype, device)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss

        print(f"  Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f} "
              f"{'✓' if improved else ''} (best: {best_val_loss:.4f})")

        model.train()

        # Clear cache once per epoch (not every iteration!)
        if use_cuda:
            torch.cuda.empty_cache()

    model.eval()
    if use_cuda:
        torch.cuda.empty_cache()

    print("\nBlock fine-tuning completed!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
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
        block_lr_multiplier=args.block_lr_multiplier,
        lora_lr_multiplier=args.lora_lr_multiplier,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_samples=args.num_samples,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation=args.gradient_accumulation,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        block_weight_decay=args.block_weight_decay,
        ln_weight_decay=args.ln_weight_decay,
        lora_weight_decay=args.lora_weight_decay,
        dataset_name=args.dataset,
        train_layernorm=args.train_layernorm,
        train_bias=args.train_bias,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        use_amp=args.use_amp,
        scheduler_type=args.scheduler,
        min_lr_ratio=args.min_lr_ratio,
        label_smoothing=args.label_smoothing,
    )

    # Save model
    print("\n" + "=" * 60)
    print("Saving fine-tuned model...")

    model.eval()
    model = model.cpu()

    # Merge LoRA weights into original modules before saving.
    # This removes LoRALayer wrappers so the saved model has no dependency
    # on LoRALayer class and can be loaded by any script.
    # For nn.Linear: W += scaling * B @ A
    # For SVDLinearWithDenseBlocks: expand SVD rank by r
    lora_merged = merge_lora_weights(model)
    if lora_merged > 0:
        print(f"  Merged {lora_merged} LoRA adapters into model weights")

    # Convert FP32 trainable params back to original dtype for saving
    save_dtype = torch.bfloat16
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
    print(f"  Converted {fp32_converted} FP32 params back to {save_dtype}")

    final_path = os.path.join(args.output_dir, "model_block_finetuned.pt")
    torch.save({'model': model, 'tokenizer': tokenizer}, final_path)
    print(f"  Saved to: {final_path}")

    # Final evaluation on test set
    if args.evaluate:
        print("\nEvaluating on test set...")
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
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--block_lr_multiplier', type=float, default=1.0,
                        help='Scale blocks LR relative to base LR')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--num_samples', type=int, default=256)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--gradient_accumulation', type=int, default=4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay for bias params (0 recommended)')
    parser.add_argument('--block_weight_decay', type=float, default=0.0,
                        help='Weight decay for blocks (0 recommended - blocks compensate SVD error)')
    parser.add_argument('--ln_weight_decay', type=float, default=0.0,
                        help='Weight decay for LayerNorm (0 recommended)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing for cross-entropy (try 0.1 for better generalization)')

    # Scheduler
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'constant'],
                        help='LR scheduler type (constant often better for fine-tuning)')
    parser.add_argument('--min_lr_ratio', type=float, default=0.1,
                        help='For cosine scheduler, minimum LR as ratio of initial (prevents decay to 0)')

    # What to train
    parser.add_argument('--train_layernorm', action='store_true', default=True)
    parser.add_argument('--no_train_layernorm', action='store_false', dest='train_layernorm')
    parser.add_argument('--train_bias', action='store_true', default=True)
    parser.add_argument('--no_train_bias', action='store_false', dest='train_bias')

    # LoRA (Low-Rank Adaptation) for additional capacity
    parser.add_argument('--use_lora', action='store_true', default=False,
                        help='Enable LoRA adapters for additional capacity')
    parser.add_argument('--no_lora', action='store_false', dest='use_lora')
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank (higher = more capacity, more params)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha scaling factor')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout rate')
    parser.add_argument('--lora_target_modules', type=str, nargs='+',
                        default=['q_proj', 'v_proj'],
                        help='Modules to apply LoRA to (default: q_proj v_proj)')
    parser.add_argument('--lora_lr_multiplier', type=float, default=1.0,
                        help='Scale LoRA LR relative to base LR')
    parser.add_argument('--lora_weight_decay', type=float, default=0.0,
                        help='Weight decay for LoRA params (0.0 recommended)')

    # AMP
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_false', dest='use_amp')

    # Data
    parser.add_argument('--dataset', type=str, default='wikitext2', choices=['wikitext2', 'c4'])

    # Evaluation
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--model_seq_len', type=int, default=2048)
    parser.add_argument('--eval_batch_size', type=int, default=1)

    args = parser.parse_args()
    main(args)
