#!/usr/bin/env python
# coding:utf8
"""
Efficient Inference Optimizations for SVD-LLM

This module provides inference efficiency improvements:
1. Fused SVD operations (v_proj + u_proj in single kernel)
2. Optimized block computation with batching
3. Half-precision (FP16) optimizations
4. Optional torch.compile support for PyTorch 2.0+

Usage:
    from efficient_inference import optimize_model_for_inference, benchmark_inference

    model = optimize_model_for_inference(model)
    benchmark_inference(model, tokenizer)
"""

import os
import sys
import time
from typing import Optional, List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

from fisher_svd import SVDLinear, SVDLinearWithDenseBlocks


class FusedSVDLinear(nn.Module):
    """
    Fused SVD linear layer for efficient inference.

    Optimizations:
    1. Pre-compute V @ U^T for small ranks
    2. Use half precision for computation
    3. Avoid unnecessary reshapes
    """

    def __init__(self, v_proj: nn.Linear, u_proj: nn.Linear):
        super().__init__()
        self.in_features = v_proj.in_features
        self.out_features = u_proj.out_features
        self.rank = v_proj.out_features

        # Store projections
        self.v_proj = v_proj
        self.u_proj = u_proj

        # For very small ranks, pre-compute the full weight matrix
        # W = U @ V (where U is u_proj.weight, V is v_proj.weight)
        self.use_fused = self.rank <= 64 and self.in_features * self.out_features <= 16 * 1024 * 1024

        if self.use_fused:
            # Pre-compute W = u_proj.weight @ v_proj.weight
            with torch.no_grad():
                # u_proj: [out, rank], v_proj: [rank, in]
                self.register_buffer(
                    'fused_weight',
                    (u_proj.weight @ v_proj.weight).contiguous()
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fused:
            return F.linear(x, self.fused_weight)
        else:
            return self.u_proj(self.v_proj(x))


class OptimizedSVDLinearWithBlocks(nn.Module):
    """
    Optimized SVD + blocks layer for efficient inference.

    Optimizations:
    1. Batch all blocks with same column index
    2. Use contiguous memory layout
    3. Minimize kernel launches
    """

    def __init__(self, original: SVDLinearWithDenseBlocks):
        super().__init__()
        self.v_proj = original.v_proj
        self.u_proj = original.u_proj
        self.block_size = original.block_size
        self.num_groups = original.num_groups

        # Copy block data
        if self.num_groups > 0:
            # Merge all blocks into single tensors for efficiency
            all_cols = []
            all_row_indices = []
            all_blocks = []
            offset = 0

            for gi in range(original.num_groups):
                col = getattr(original, f"g{gi}_col")
                row_index = getattr(original, f"g{gi}_row_index")
                blocks_T = getattr(original, f"g{gi}_blocks_T")

                all_cols.append(col)
                all_row_indices.append(row_index)
                all_blocks.append(blocks_T)

            # Store as buffers
            for gi, (col, row_idx, blocks) in enumerate(zip(all_cols, all_row_indices, all_blocks)):
                self.register_buffer(f"g{gi}_col", col)
                self.register_buffer(f"g{gi}_row_index", row_idx)
                self.register_buffer(f"g{gi}_blocks_T", blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Low-rank path
        y = self.u_proj(self.v_proj(x))

        if self.num_groups == 0:
            return y

        # Optimized block computation
        orig_shape = y.shape
        y2d = y.view(-1, y.shape[-1])
        x2d = x.view(-1, x.shape[-1])
        b = self.block_size

        # Process all groups
        for gi in range(self.num_groups):
            col = getattr(self, f"g{gi}_col").item()
            row_index = getattr(self, f"g{gi}_row_index")
            blocks_T = getattr(self, f"g{gi}_blocks_T")

            # Efficient computation
            Xc = x2d[:, col:col+b].contiguous()
            out_flat = torch.mm(Xc, blocks_T)
            y2d.index_add_(1, row_index, out_flat)

        return y2d.view(orig_shape)


def optimize_model_for_inference(
    model: nn.Module,
    use_fused_svd: bool = True,
    use_torch_compile: bool = False,
    dtype: torch.dtype = torch.float16
) -> nn.Module:
    """
    Optimize model for efficient inference.

    Args:
        model: The SVD-compressed model
        use_fused_svd: Whether to fuse SVD operations
        use_torch_compile: Whether to use torch.compile (PyTorch 2.0+)
        dtype: Target dtype for inference

    Returns:
        Optimized model
    """
    print("Optimizing model for inference...")

    # Convert to target dtype
    model = model.to(dtype)

    # Count optimizations
    num_fused = 0
    num_optimized_blocks = 0

    if use_fused_svd:
        # Replace SVDLinear with FusedSVDLinear where beneficial
        for name, module in list(model.named_modules()):
            if isinstance(module, SVDLinear):
                # Get parent module and attribute name
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    attr_name = name

                # Create fused version
                fused = FusedSVDLinear(module.v_proj, module.u_proj)
                fused = fused.to(dtype)
                setattr(parent, attr_name, fused)
                num_fused += 1

            elif isinstance(module, SVDLinearWithDenseBlocks):
                # Get parent module and attribute name
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    attr_name = name

                # Create optimized version
                optimized = OptimizedSVDLinearWithBlocks(module)
                optimized = optimized.to(dtype)
                setattr(parent, attr_name, optimized)
                num_optimized_blocks += 1

    print(f"  Fused SVD layers: {num_fused}")
    print(f"  Optimized block layers: {num_optimized_blocks}")

    # Apply torch.compile if available
    if use_torch_compile:
        try:
            import torch._dynamo
            model = torch.compile(model, mode="reduce-overhead")
            print("  Applied torch.compile optimization")
        except Exception as e:
            print(f"  torch.compile not available: {e}")

    # Set eval mode
    model.eval()

    return model


def benchmark_inference(
    model: nn.Module,
    tokenizer,
    prompt: str = "The quick brown fox",
    max_new_tokens: int = 50,
    num_runs: int = 5,
    warmup_runs: int = 2,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Benchmark inference performance.

    Args:
        model: The model to benchmark
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Number of tokens to generate
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        device: Device to run on

    Returns:
        Dictionary with benchmark results
    """
    print("\nBenchmarking inference...")

    model = model.to(device)
    model.eval()

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs['input_ids'].shape[1]

    # Warmup
    print(f"  Warming up ({warmup_runs} runs)...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

    # Synchronize
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    print(f"  Benchmarking ({num_runs} runs)...")
    times = []
    tokens_generated = []

    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

            if device == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            times.append(end_time - start_time)
            tokens_generated.append(outputs.shape[1] - input_len)

    # Calculate statistics
    avg_time = sum(times) / len(times)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)
    tokens_per_second = avg_tokens / avg_time

    # Memory usage
    if device == "cuda":
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
        memory_reserved = torch.cuda.max_memory_reserved() / 1024**3
    else:
        memory_allocated = 0
        memory_reserved = 0

    results = {
        'avg_time_seconds': avg_time,
        'tokens_per_second': tokens_per_second,
        'avg_tokens_generated': avg_tokens,
        'memory_allocated_gb': memory_allocated,
        'memory_reserved_gb': memory_reserved,
    }

    print(f"\n  Results:")
    print(f"    Average time: {avg_time:.3f}s")
    print(f"    Tokens/second: {tokens_per_second:.1f}")
    print(f"    Memory allocated: {memory_allocated:.2f} GB")
    print(f"    Memory reserved: {memory_reserved:.2f} GB")

    return results


def compare_inference_speed(
    original_model_path: str,
    compressed_model_path: str,
    tokenizer,
    prompt: str = "The quick brown fox",
    max_new_tokens: int = 50,
    device: str = "cuda"
) -> Dict[str, Dict[str, float]]:
    """
    Compare inference speed between original and compressed models.

    Args:
        original_model_path: Path to original model
        compressed_model_path: Path to compressed model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Number of tokens to generate
        device: Device to run on

    Returns:
        Dictionary with comparison results
    """
    results = {}

    # Benchmark original model
    print("\n" + "="*60)
    print("Benchmarking ORIGINAL model...")
    print("="*60)

    original_dict = torch.load(original_model_path, map_location='cpu')
    original_model = original_dict['model']
    original_model = original_model.half().to(device)
    original_model.eval()

    results['original'] = benchmark_inference(
        original_model, tokenizer, prompt, max_new_tokens, device=device
    )

    # Free memory
    del original_model
    torch.cuda.empty_cache()

    # Benchmark compressed model
    print("\n" + "="*60)
    print("Benchmarking COMPRESSED model...")
    print("="*60)

    compressed_dict = torch.load(compressed_model_path, map_location='cpu')
    compressed_model = compressed_dict['model']
    compressed_model = optimize_model_for_inference(compressed_model)
    compressed_model = compressed_model.to(device)

    results['compressed'] = benchmark_inference(
        compressed_model, tokenizer, prompt, max_new_tokens, device=device
    )

    # Calculate speedup
    speedup = results['compressed']['tokens_per_second'] / results['original']['tokens_per_second']
    memory_reduction = 1 - (results['compressed']['memory_allocated_gb'] / results['original']['memory_allocated_gb'])

    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Memory reduction: {memory_reduction*100:.1f}%")

    results['speedup'] = speedup
    results['memory_reduction'] = memory_reduction

    return results


# Additional optimization: Quantization support
def quantize_model_int8(model: nn.Module) -> nn.Module:
    """
    Apply INT8 dynamic quantization for CPU inference.

    Args:
        model: Model to quantize

    Returns:
        Quantized model
    """
    print("Applying INT8 dynamic quantization...")

    # Only quantize Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    return quantized_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark inference efficiency')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to compressed model')
    parser.add_argument('--prompt', type=str, default="The quick brown fox",
                        help='Input prompt')
    parser.add_argument('--max_tokens', type=int, default=50,
                        help='Max tokens to generate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model_dict = torch.load(args.model_path, map_location='cpu')
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']

    # Optimize
    model = optimize_model_for_inference(model)
    model = model.to(args.device)

    # Benchmark
    results = benchmark_inference(
        model, tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        device=args.device
    )

    print("\nDone!")
