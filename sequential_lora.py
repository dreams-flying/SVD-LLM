#!/usr/bin/env python
# coding:utf8
"""
Sequential Low-rank Approximation (LoRA) Fine-tuning for SVD-LLM

Based on SVD-LLM paper Section 3.2:
- Sequential training: First train W'_u, then train W'_v
- This avoids interference between the two low-rank matrices
- Finally merge: W'_u ← W'_u + B_u × A_u, W'_v ← W'_v + B_v × A_v

For Fisher SVD compressed models:
- Original linear layers (e.g., q_proj) are replaced with SVDLinear/SVDLinearWithDenseBlocks
- Each SVDLinear contains: v_proj (input side) and u_proj (output side)
- Module paths like: model.layers.0.self_attn.q_proj.v_proj

Usage:
    python sequential_lora.py --prune_model path/to/compressed_model.pt \
        --output_dir ./lora_output \
        --num_epochs 2 \
        --learning_rate 3e-4

Reference: https://github.com/tloen/alpaca-lora/blob/main/finetune.py
"""

import os
import sys
import argparse
from typing import List, Optional

import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from tqdm import tqdm

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from Prompter import Prompter, ZeroPrompter

device = "cuda" if torch.cuda.is_available() else "cpu"


def wikitext2():
    """Load wikitext2 dataset."""
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    return traindata, testdata


def ptb():
    """Load PTB dataset."""
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    return traindata, valdata


def get_target_modules_for_phase(phase: int) -> List[str]:
    """
    Get target modules for each phase of sequential LoRA training.

    For Fisher SVD compressed models:
    - Original linear layers (e.g., q_proj) are replaced with SVDLinear/SVDLinearWithDenseBlocks
    - Each SVDLinear contains: v_proj (input side) and u_proj (output side)
    - Module paths like: model.layers.0.self_attn.q_proj.v_proj

    Phase 1: Train U projections (output side of SVD decomposition)
    Phase 2: Train V projections (input side of SVD decomposition)

    Args:
        phase: 1 for U projections, 2 for V projections

    Returns:
        List of module names to target
    """
    if phase == 1:
        # Phase 1: Train U projections (freeze V projections)
        # This targets all u_proj layers inside SVDLinear modules
        return ["u_proj"]
    else:
        # Phase 2: Train V projections (freeze U projections)
        # This targets all v_proj layers inside SVDLinear modules
        return ["v_proj"]


def merge_lora_weights(model):
    """
    Merge LoRA weights into the base model.

    This performs: W ← W + B × A
    where B and A are the LoRA adaptation matrices.
    """
    print("  Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    return model


def sequential_lora_finetune(
    model,
    tokenizer,
    phase: int,
    batch_size: int = 64,
    micro_batch_size: int = 4,
    cutoff_len: int = 256,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    val_set_size: int = 2000,
    data_path: str = "yahma/alpaca-cleaned",
    num_epochs: int = 2,
    learning_rate: float = 3e-4,
    output_dir: str = "Checkpoints/tune",
    use_wikitext: bool = False,
):
    """
    Apply LoRA fine-tuning for one phase of sequential training.

    Args:
        model: The model to fine-tune
        tokenizer: The tokenizer
        phase: 1 for U projections, 2 for V projections
        Other args: Standard LoRA training hyperparameters

    Returns:
        Fine-tuned model with LoRA weights merged
    """
    phase_name = "U projections" if phase == 1 else "V projections"
    print(f"\n{'='*60}")
    print(f"Phase {phase}: Fine-tuning {phase_name}")
    print(f"{'='*60}")

    gradient_accumulation_steps = batch_size // micro_batch_size
    prompter = ZeroPrompter()

    if device == 'cuda':
        model.half()

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt

    def split_and_tokenizer(test_data, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(test_data[field_name]), return_tensors='pt').input_ids[0]
        nsamples = test_ids.numel() // seq_len

        test_set = []
        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_set.append({
                'input_ids': batch,
                'labels': batch
            })
        return test_set

    # Prepare model for LoRA (only needed once, skip if already prepared)
    # Check if model was already prepared by looking for CastOutputToFloat wrapper
    try:
        # Try to access the output embedding weight - if it fails, model is already prepared
        if hasattr(model, 'get_output_embeddings'):
            output_emb = model.get_output_embeddings()
            if output_emb is not None and hasattr(output_emb, 'weight'):
                model = prepare_model_for_int8_training(model)
                print(f"  Model prepared for training")
            else:
                print(f"  Model already prepared, skipping prepare_model_for_int8_training")
        else:
            model = prepare_model_for_int8_training(model)
    except Exception as e:
        print(f"  Skipping prepare_model_for_int8_training: {e}")

    # Get target modules for this phase
    target_modules = get_target_modules_for_phase(phase)
    print(f"  Target modules: {target_modules}")

    # Check which target modules actually exist in the model
    available_modules = set()
    for name, _ in model.named_modules():
        for target in target_modules:
            if name.endswith(target):
                available_modules.add(target)

    if not available_modules:
        print(f"  Warning: No target modules found for phase {phase}")
        print(f"  Available module names (sample):")
        for i, (name, _) in enumerate(model.named_modules()):
            if i < 30:
                print(f"    {name}")
        return model

    print(f"  Found modules: {list(available_modules)}")

    # PEFT compatibility: Add bias attribute to SVDLinear modules if missing
    # PEFT's _find_and_replace checks for bias attribute on target modules
    from fisher_svd import SVDLinear, SVDLinearWithDenseBlocks
    for name, module in model.named_modules():
        if isinstance(module, (SVDLinear, SVDLinearWithDenseBlocks)):
            if not hasattr(module, 'bias'):
                module.bias = None

    # Configure LoRA
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=list(available_modules),
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Load training data
    if use_wikitext:
        train_data, _ = wikitext2()
        train_data = split_and_tokenizer(train_data, tokenizer, cutoff_len, 'text')
        val_data = {"wikitext2": split_and_tokenizer(wikitext2()[1], tokenizer, 128, 'text')}
    else:
        data = load_dataset(data_path)
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = {
            data_path: train_val["test"].shuffle().map(generate_and_tokenize_prompt),
        }

    # Create phase-specific output directory
    phase_output_dir = os.path.join(output_dir, f"phase{phase}")
    os.makedirs(phase_output_dir, exist_ok=True)

    # Training
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            logging_first_step=True,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=200,
            output_dir=phase_output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=None,
            group_by_length=False,
            report_to="none",
            run_name=f"sequential_lora_phase{phase}",
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    trainer.train()
    model.state_dict = old_state_dict

    # Merge LoRA weights into base model
    print(f"\n  Merging Phase {phase} LoRA weights...")
    model = merge_lora_weights(model)

    return model


def main(args):
    """
    Main function for sequential LoRA fine-tuning.

    Implements the Sequential Low-rank Approximation strategy from SVD-LLM:
    1. First freeze V projections and fine-tune U projections with LoRA
    2. Then freeze U projections and fine-tune V projections with LoRA
    3. Merge LoRA weights after each phase
    """
    print("="*60)
    print("Sequential Low-rank Approximation (LoRA) Fine-tuning")
    print("="*60)
    print(f"  Input model: {args.prune_model}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  LoRA rank: {args.lora_r}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs per phase: {args.num_epochs}")

    # Load compressed model
    print("\nLoading compressed model...")
    pruned_dict = torch.load(args.prune_model, map_location='cpu')
    tokenizer = pruned_dict['tokenizer']
    model = pruned_dict['model']

    # Move to device
    if device == 'cuda':
        model = model.cuda()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Phase 1: Fine-tune U projections
    if not args.skip_phase1:
        model = sequential_lora_finetune(
            model=model,
            tokenizer=tokenizer,
            phase=1,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            cutoff_len=args.cutoff_len,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            val_set_size=args.val_set_size,
            data_path=args.data_path,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            use_wikitext=args.use_wikitext,
        )

        # Save intermediate model
        if args.save_intermediate:
            print("\nSaving model after Phase 1...")
            torch.save(
                {'model': model, 'tokenizer': tokenizer},
                os.path.join(args.output_dir, "model_after_phase1.pt")
            )

    # Phase 2: Fine-tune V projections
    if not args.skip_phase2:
        model = sequential_lora_finetune(
            model=model,
            tokenizer=tokenizer,
            phase=2,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            cutoff_len=args.cutoff_len,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            val_set_size=args.val_set_size,
            data_path=args.data_path,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            use_wikitext=args.use_wikitext,
        )

    # Save final model
    print("\n" + "="*60)
    print("Saving final model...")
    final_path = os.path.join(args.output_dir, "model_final.pt")
    torch.save({'model': model, 'tokenizer': tokenizer}, final_path)
    print(f"  Saved to: {final_path}")

    # Evaluate final model
    if args.evaluate:
        print("\nEvaluating final model...")
        from utils.eval_utils import ppl_eval

        model = model.float().to(device)
        ppl_eval(
            model, tokenizer,
            datasets=['wikitext2'],
            model_seq_len=args.model_seq_len,
            batch_size=args.eval_batch_size,
            device=device
        )

    print("\n" + "="*60)
    print("Sequential LoRA fine-tuning completed!")
    print("="*60)

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequential LoRA Fine-tuning for SVD-LLM')

    # Model paths
    parser.add_argument('--prune_model', type=str, required=True,
                        help='Path to compressed model (.pt file)')
    parser.add_argument('--output_dir', type=str, default='./lora_output',
                        help='Output directory for checkpoints')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Total batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4,
                        help='Micro batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of epochs per phase')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--cutoff_len', type=int, default=256,
                        help='Maximum sequence length')
    parser.add_argument('--val_set_size', type=int, default=2000,
                        help='Validation set size')

    # LoRA configuration
    parser.add_argument('--lora_r', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout')

    # Data
    parser.add_argument('--data_path', type=str, default='yahma/alpaca-cleaned',
                        help='Training data path')
    parser.add_argument('--use_wikitext', action='store_true',
                        help='Use wikitext2 for training instead of instruction data')

    # Phase control
    parser.add_argument('--skip_phase1', action='store_true',
                        help='Skip Phase 1 (U projection training)')
    parser.add_argument('--skip_phase2', action='store_true',
                        help='Skip Phase 2 (V projection training)')
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Save model after each phase')

    # Evaluation
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate model after training')
    parser.add_argument('--model_seq_len', type=int, default=2048,
                        help='Model sequence length for evaluation')
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help='Batch size for evaluation')

    args = parser.parse_args()
    main(args)
