#coding:utf8
"""
Fisher-Aware SVD Truncation

This module implements the Fisher-Aware SVD truncation algorithm for LLM compression.
The algorithm uses empirical Fisher information as a Hessian approximation to determine
which singular values are most important for the task loss, enabling end-to-end
task-aware compression.

Key idea: Instead of truncating based solely on singular value magnitude (like standard SVD),
we compute importance scores S_i = σ_i² × F_ii, where F_ii is the Fisher information
(squared gradient) of the loss with respect to σ_i.

Reference: Second-order sensitivity analysis for neural network compression.
"""

import os
import sys
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from utils.data_utils import get_calib_train_data, get_loaders
from utils.model_utils import find_layers, get_model_from_huggingface
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_opt import SVDOPTDecoderLayer


class SVDParameterizedLinear(nn.Module):
    """
    A linear layer parameterized by its SVD components: W = U @ diag(sigma) @ V^T

    This allows gradients to flow through the singular values, enabling
    Fisher information estimation for each singular direction.
    """

    def __init__(self, U: torch.Tensor, sigma: torch.Tensor, VT: torch.Tensor,
                 bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.U = nn.Parameter(U, requires_grad=False)  # Frozen
        self.sigma = nn.Parameter(sigma, requires_grad=True)  # Differentiable
        self.VT = nn.Parameter(VT, requires_grad=False)  # Frozen
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # W = U @ diag(sigma) @ V^T
        # output = x @ W^T = x @ V @ diag(sigma) @ U^T

        # Store original dtype for output
        original_dtype = x.dtype

        # Convert input to float32 for numerical stability and gradient computation
        # sigma needs to be in float32 for gradient flow
        x = x.float()

        out = torch.matmul(x, self.VT.T)  # x @ V
        out = out * self.sigma  # element-wise multiply with sigma (gradients flow through here)
        out = torch.matmul(out, self.U.T)  # @ U^T
        if self.bias is not None:
            out = out + self.bias

        # Convert back to original dtype
        return out.to(original_dtype)


class SVDLinear(nn.Module):
    """
    A linear layer factorized as W = U @ V where U and V are low-rank matrices.
    Used for applying SVD compression to linear layers.
    """

    def __init__(self, v_proj: nn.Linear, u_proj: nn.Linear):
        super().__init__()
        self.v_proj = v_proj
        self.u_proj = u_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.u_proj(self.v_proj(x))


class SVDLinearWithDenseBlocks(nn.Module):
    """
    Low-rank SVD + sparse dense blocks for better approximation.

    W ≈ U_k @ Σ_k @ V_k^T + R

    where R is represented as a list of dense blocks (not sparse tensor).
    Uses bucket-based GEMM + index_add for efficient inference.

    Key insight: Traditional SVD has k(m+n) params, which can exceed mn for large k.
    This approach uses small k + critical dense blocks to get better accuracy
    with controlled parameter budget.
    """

    def __init__(self, v_proj: nn.Linear, u_proj: nn.Linear,
                 block_size: int = 16,
                 groups: Optional[List[Dict]] = None):
        """
        Args:
            v_proj: V projection (in_features -> rank)
            u_proj: U projection (rank -> out_features)
            block_size: Size of dense blocks (default 16)
            groups: Pre-packed block groups by col_start, each containing:
                    {"col": int, "row_index": LongTensor, "blocks_T": Tensor}
        """
        super().__init__()
        self.v_proj = v_proj
        self.u_proj = u_proj
        self.block_size = block_size
        self.num_groups = 0

        if groups is not None:
            self.num_groups = len(groups)
            # Register group tensors as buffers for proper device handling
            for gi, g in enumerate(groups):
                self.register_buffer(f"g{gi}_col", torch.tensor(g["col"], dtype=torch.long))
                self.register_buffer(f"g{gi}_row_index", g["row_index"])
                self.register_buffer(f"g{gi}_blocks_T", g["blocks_T"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Low-rank path: same as SVDLinear
        y = self.u_proj(self.v_proj(x))

        # Residual path: bucket-based dense GEMM + index_add
        if self.num_groups == 0:
            return y

        # Reshape for 2D operations
        orig_shape = y.shape
        y2d = y.reshape(-1, y.shape[-1])  # [B*S, out]
        x2d = x.reshape(-1, x.shape[-1])  # [B*S, in]
        b = self.block_size

        for gi in range(self.num_groups):
            col = getattr(self, f"g{gi}_col").item()
            row_index = getattr(self, f"g{gi}_row_index")  # [L]
            blocks_T = getattr(self, f"g{gi}_blocks_T")    # [b, L]

            # Extract input slice for this column bucket
            Xc = x2d[:, col:col+b]  # [B*S, b]

            # Single GEMM for all blocks in this bucket
            out_flat = torch.matmul(Xc, blocks_T)  # [B*S, L]

            # Scatter-add to output
            y2d.index_add_(1, row_index, out_flat)

        return y2d.reshape(orig_shape)

    @staticmethod
    def pack_blocks_by_col(blocks: List[Dict], block_size: int, device, dtype=None) -> List[Dict]:
        """
        Pack blocks into column-bucket format for efficient inference.

        Args:
            blocks: List of {"row": int, "col": int, "val": Tensor[b,b]}
            block_size: Block size
            device: Target device
            dtype: Target dtype (e.g., float16, bfloat16). If None, keeps original dtype.

        Returns:
            List of packed groups: {"col": int, "row_index": Tensor, "blocks_T": Tensor}
        """
        from collections import defaultdict

        buckets = defaultdict(list)
        for blk in blocks:
            buckets[blk["col"]].append(blk)

        groups = []
        b = block_size
        ar = torch.arange(b, device=device)

        for col, blks in sorted(buckets.items()):
            # Build row_index: [g*b] where g = number of blocks in bucket
            rows = torch.tensor([blk["row"] for blk in blks], device=device)
            row_index = (rows[:, None] + ar[None, :]).reshape(-1).long()

            # Build blocks_T: [b, g*b]
            # Each block's transpose is concatenated horizontally
            # Convert to target dtype for proper matmul with model activations
            if dtype is not None:
                blocks_T = torch.cat([blk["val"].to(device=device, dtype=dtype).T for blk in blks], dim=1)
            else:
                blocks_T = torch.cat([blk["val"].to(device).T for blk in blks], dim=1)

            groups.append({
                "col": col,
                "row_index": row_index,
                "blocks_T": blocks_T
            })

        return groups


def select_residual_blocks(W: torch.Tensor, U: torch.Tensor, S: torch.Tensor,
                           VT: torch.Tensor, block_size: int, budget_blocks: int,
                           row_importance: Optional[torch.Tensor] = None,
                           col_importance: Optional[torch.Tensor] = None,
                           top_per_row: int = 8,
                           layer_factor: float = 1.0,
                           use_fisher_weight: bool = True) -> List[Dict]:
    """
    Select high-importance residual blocks for W - U @ diag(S) @ VT.

    Fully optimized implementation:
    - One GEMM per row-tile (not per block)
    - No col-loop: reshape + batch energy computation
    - No .item() in loops (avoid GPU sync)
    - Clone only top_per_row blocks (not all blocks then filter)
    - Stores actual row_end/col_end for edge blocks
    - Uses BOTH row and column importance for scoring

    Args:
        W: Original weight matrix [out, in]
        U: Left singular vectors [out, k]
        S: Singular values [k]
        VT: Right singular vectors transposed [k, in]
        block_size: Size of blocks
        budget_blocks: Maximum number of blocks to select
        row_importance: Per-output importance weights [out] (Fisher row)
        col_importance: Per-input importance weights [in] (Fisher col)
        top_per_row: Max candidates per row-tile to reduce search space
        layer_factor: Multiplier for scores (for cross-layer balancing)
        use_fisher_weight: If False, only use Frobenius norm (for debugging)

    Returns:
        List of {"row": int, "col": int, "row_end": int, "col_end": int,
                 "val": Tensor[b,b], "score": float}
        NOTE: Returns score so Phase3b can use Fisher-weighted scores directly
    """
    m, n = W.shape
    b = block_size
    device = W.device
    orig_dtype = W.dtype

    # Number of row/col tiles
    n_row_tiles = (m + b - 1) // b
    n_col_tiles = (n + b - 1) // b

    # Padded dimensions (for reshape)
    n_padded = n_col_tiles * b

    # Pre-compute US = U @ diag(S) for efficiency (in float32 for stability)
    US = (U * S).float()  # [m, k]
    VT_f = VT.float()
    W_f = W.float()

    # Pre-compute importance weights per row-tile and col-tile
    if use_fisher_weight and row_importance is not None:
        imp_row = row_importance.float().to(device)
        row_weights = torch.zeros(n_row_tiles, device=device, dtype=torch.float32)
        for ri in range(n_row_tiles):
            row_start = ri * b
            row_end = min(row_start + b, m)
            row_weights[ri] = imp_row[row_start:row_end].mean()
    else:
        row_weights = torch.ones(n_row_tiles, device=device, dtype=torch.float32)

    if use_fisher_weight and col_importance is not None:
        imp_col = col_importance.float().to(device)
        col_weights = torch.zeros(n_col_tiles, device=device, dtype=torch.float32)
        for ci in range(n_col_tiles):
            col_start = ci * b
            col_end = min(col_start + b, n)
            col_weights[ci] = imp_col[col_start:col_end].mean()
    else:
        col_weights = torch.ones(n_col_tiles, device=device, dtype=torch.float32)

    candidates = []

    for ri in range(n_row_tiles):
        row_start = ri * b
        row_end = min(row_start + b, m)
        actual_row_size = row_end - row_start

        # === ONE GEMM per row-tile: compute entire row's approximation ===
        US_row = US[row_start:row_end, :]  # [actual_row_size, k]
        W_approx_row = US_row @ VT_f  # [actual_row_size, n]
        W_row = W_f[row_start:row_end, :]

        # Residual row
        R_row = W_row - W_approx_row  # [actual_row_size, n]

        # === NO COL-LOOP: Reshape to compute all block energies at once ===
        # Pad to [b, n_padded] for clean reshape
        R_row_padded = torch.zeros(b, n_padded, device=device, dtype=torch.float32)
        R_row_padded[:actual_row_size, :n] = R_row

        # Reshape: [b, n_col_tiles, b] -> permute -> [n_col_tiles, b, b]
        R_blocks = R_row_padded.view(b, n_col_tiles, b).permute(1, 0, 2)

        # Compute energy for all blocks at once: [n_col_tiles]
        block_energies = (R_blocks ** 2).sum(dim=(1, 2))

        # Apply importance weights: geometric mean of row and col importance
        # Score = ||R||_F^2 * sqrt(row_weight * col_weight) * layer_factor
        combined_weights = torch.sqrt(row_weights[ri] * col_weights) * layer_factor
        weighted_energies = block_energies * combined_weights

        # === Select top_per_row using torch.topk ===
        k = min(top_per_row, n_col_tiles)
        top_scores, top_indices = torch.topk(weighted_energies, k)

        # === Clone ONLY the top_per_row blocks ===
        top_scores_cpu = top_scores.cpu().tolist()
        top_indices_cpu = top_indices.cpu().tolist()

        for score, ci in zip(top_scores_cpu, top_indices_cpu):
            if score <= 0:
                continue

            col_start = ci * b
            col_end = min(col_start + b, n)

            # Extract the residual block (already padded in R_blocks)
            R_block = R_blocks[ci].clone()

            candidates.append({
                "score": score,
                "row": row_start,
                "col": col_start,
                "row_end": row_end,
                "col_end": col_end,
                "val": R_block.to(orig_dtype).cpu()  # Back to orig dtype, move to CPU
            })

    # Global selection: top budget_blocks by score
    candidates.sort(key=lambda x: -x["score"])
    selected = candidates[:budget_blocks]

    return selected


class SVDLinearTrainable(nn.Module):
    """
    A trainable low-rank linear layer for Phase 5 distillation fine-tuning.

    W = U @ diag(S) @ VT where U, S, VT are all trainable parameters.
    The rank is fixed, but values can be optimized to minimize distillation loss.

    This is essentially a LoRA-like structure, but initialized from SVD instead of random.
    """

    def __init__(self, U: torch.Tensor, S: torch.Tensor, VT: torch.Tensor,
                 bias: Optional[torch.Tensor] = None, train_bias: bool = False):
        super().__init__()
        # U: (out_features, rank), S: (rank,), VT: (rank, in_features)
        self.U = nn.Parameter(U.float())
        self.S = nn.Parameter(S.float())
        self.VT = nn.Parameter(VT.float())

        if bias is not None:
            self.bias = nn.Parameter(bias.float(), requires_grad=train_bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # W = U @ diag(S) @ VT
        # Forward: x @ W^T = x @ VT^T @ diag(S) @ U^T
        original_dtype = x.dtype
        x = x.float()

        # Efficient computation: x @ V @ diag(S) @ U^T
        out = torch.matmul(x, self.VT.T)  # x @ V: (batch, seq, rank)
        out = out * self.S                 # element-wise: (batch, seq, rank)
        out = torch.matmul(out, self.U.T)  # @ U^T: (batch, seq, out_features)

        if self.bias is not None:
            out = out + self.bias

        return out.to(original_dtype)

    @torch.no_grad()
    def get_weight(self) -> torch.Tensor:
        """Reconstruct the full weight matrix W = U @ diag(S) @ VT"""
        return (self.U * self.S) @ self.VT

    @torch.no_grad()
    def merge_to_linear(self) -> nn.Linear:
        """Convert to a standard nn.Linear for inference efficiency."""
        weight = self.get_weight()
        out_features, in_features = self.U.shape[0], self.VT.shape[1]
        linear = nn.Linear(in_features, out_features, bias=self.bias is not None)
        linear.weight.data = weight.to(linear.weight.dtype)
        if self.bias is not None:
            linear.bias.data = self.bias.to(linear.bias.dtype)
        return linear

    def get_svd_components(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Return current U, S, VT, bias values (detached, on CPU)."""
        return (
            self.U.data.cpu(),
            self.S.data.cpu(),
            self.VT.data.cpu(),
            self.bias.data.cpu() if self.bias is not None else None
        )


class FisherAwareSVD:
    """
    Fisher-Aware SVD compression for LLMs.

    This class implements the three-phase algorithm:
    1. Phase 1: SVD decomposition of each linear layer
    2. Phase 2: Sensitivity estimation via empirical Fisher information
    3. Phase 3: Global truncation based on importance scores
    """

    def __init__(self, model: nn.Module, model_name: str, device: str = "cuda",
                 num_gpus: int = 1):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus

        # Determine available GPUs
        if num_gpus > 1 and torch.cuda.device_count() >= num_gpus:
            self.devices = [f"cuda:{i}" for i in range(num_gpus)]
            self.use_multi_gpu = True
            print(f"  Using {num_gpus} GPUs: {self.devices}")
        else:
            self.devices = [device]
            self.use_multi_gpu = False

        # Get layers based on model type
        if "opt" in model_name:
            self.layers = model.model.decoder.layers
        else:
            self.layers = model.model.layers

        # Storage for SVD components and Fisher information
        self.svd_components: Dict[str, Dict[str, Tuple[torch.Tensor, ...]]] = {}
        self.fisher_info: Dict[str, Dict[str, torch.Tensor]] = {}
        self.original_layers: Dict[str, nn.Module] = {}

    def phase1_svd_decomposition(self, whitening_mat: Optional[Dict] = None,
                                    store_original_weights: bool = False) -> None:
        """
        Phase 1: Perform SVD decomposition on each linear layer.

        Args:
            whitening_mat: Optional whitening matrices from SVD-LLM profiling.
                          If provided, applies whitening before SVD.
            store_original_weights: If True, store original weights for residual block selection
        """
        print("Phase 1: SVD Decomposition...")

        # Initialize storage for original weights if needed (for residual block selection)
        if store_original_weights:
            self.original_weights = {}

        for layer_idx in tqdm(range(len(self.layers))):
            layer = self.layers[layer_idx]
            subset = find_layers(layer)

            layer_svd = {}
            if store_original_weights:
                self.original_weights[layer_idx] = {}

            for name, module in subset.items():
                W = module.weight.data.float()

                # Store original weight if requested (for residual block selection)
                if store_original_weights:
                    self.original_weights[layer_idx][name] = W.cpu().clone()

                # Apply whitening if available
                if whitening_mat is not None and layer_idx in whitening_mat:
                    if name in whitening_mat[layer_idx]:
                        scaling_diag_matrix = whitening_mat[layer_idx][name].to(W.device)
                        try:
                            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                        except:
                            scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(W.device)
                            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                        W_scale = torch.matmul(W, scaling_diag_matrix.float())
                        U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
                        VT = torch.matmul(VT, scaling_matrix_inv.float())
                    else:
                        U, S, VT = torch.linalg.svd(W, full_matrices=False)
                else:
                    U, S, VT = torch.linalg.svd(W, full_matrices=False)

                # Store bias if present
                bias = module.bias.data.clone() if module.bias is not None else None

                layer_svd[name] = (U.cpu(), S.cpu(), VT.cpu(), bias.cpu() if bias is not None else None)

            self.svd_components[layer_idx] = layer_svd

        print(f"  Decomposed {len(self.layers)} layers")

    def _replace_with_svd_layers(self) -> None:
        """Replace original linear layers with SVD-parameterized layers."""
        # Store references to SVD layers for later access
        self.svd_layer_refs = {}

        for layer_idx in range(len(self.layers)):
            layer = self.layers[layer_idx]
            subset = find_layers(layer)

            self.svd_layer_refs[layer_idx] = {}

            for name, module in subset.items():
                if layer_idx in self.svd_components and name in self.svd_components[layer_idx]:
                    U, S, VT, bias = self.svd_components[layer_idx][name]
                    svd_layer = SVDParameterizedLinear(
                        U.to(self.device),
                        S.to(self.device),
                        VT.to(self.device),
                        bias.to(self.device) if bias is not None else None
                    )

                    # Store reference for Fisher estimation
                    self.svd_layer_refs[layer_idx][name] = svd_layer

                    # Replace the layer
                    self._set_module_by_name(layer, name, svd_layer)

    def _restore_original_layers(self) -> None:
        """Restore original linear layers from SVD components."""
        dtype = next(iter(self.model.parameters())).dtype

        for layer_idx in range(len(self.layers)):
            layer = self.layers[layer_idx]

            # Use stored references instead of find_layers
            if layer_idx not in self.svd_layer_refs:
                continue

            for name, svd_layer in self.svd_layer_refs[layer_idx].items():
                if layer_idx in self.svd_components and name in self.svd_components[layer_idx]:
                    U, S, VT, bias = self.svd_components[layer_idx][name]
                    # Reconstruct W = U @ diag(S) @ VT
                    W = torch.matmul(U * S, VT)

                    # Create new linear layer with same dtype as SVD layer
                    out_features, in_features = W.shape
                    new_linear = nn.Linear(in_features, out_features, bias=bias is not None)
                    new_linear.weight.data = W.to(dtype)
                    if bias is not None:
                        new_linear.bias.data = bias.to(dtype)

                    self._set_module_by_name(layer, name, new_linear.to(self.device))

    def _set_module_by_name(self, parent: nn.Module, name: str, new_module: nn.Module) -> None:
        """Set a submodule by its name path."""
        parts = name.split('.')
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def phase2_sensitivity_estimation(self, calib_loader: List[Dict],
                                       use_low_resource: bool = True) -> None:
        """
        Phase 2: Estimate sensitivity using empirical Fisher information.

        For each singular value σ_i, we compute:
            F_ii = E_x[(∂L/∂σ_i)²]

        Args:
            calib_loader: Calibration data loader
            use_low_resource: If True, process layer by layer to save memory
        """
        print("Phase 2: Sensitivity Estimation via Empirical Fisher...")

        if use_low_resource:
            self._estimate_fisher_low_resource(calib_loader)
        else:
            self._estimate_fisher_full(calib_loader)

    def _estimate_fisher_low_resource(self, calib_loader: List[Dict]) -> None:
        """
        Low-resource Fisher estimation using proxy loss.

        Memory-efficient approach using layer-local proxy loss that approximates
        the sensitivity of each singular value to the overall task loss.

        Proxy loss = ||output - original_output||² + 0.1 * Var(output - original_output)

        This captures how much each singular value affects the layer output,
        which is proportional to its effect on the final task loss.
        """

        print("  Using layer-wise estimation with proxy loss...")

        # Move embedding layers to device
        if "opt" in self.model_name:
            self.model.model.decoder.embed_tokens = self.model.model.decoder.embed_tokens.to(self.device)
            self.model.model.decoder.embed_positions = self.model.model.decoder.embed_positions.to(self.device)
        else:
            self.model.model.embed_tokens = self.model.model.embed_tokens.to(self.device)

        # Capture inputs to first layer
        dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros(
            (len(calib_loader), self.model.seqlen, self.model.config.hidden_size),
            dtype=dtype, device=self.device
        )
        input_ids_list = []
        cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                if cache['attention_mask'] is None:
                    cache['attention_mask'] = kwargs['attention_mask']
                    if 'position_ids' in kwargs:
                        cache['position_ids'] = kwargs['position_ids']
                else:
                    cache['attention_mask'] = torch.cat(
                        (cache['attention_mask'], kwargs['attention_mask']), dim=0
                    )
                    if 'position_ids' in kwargs:
                        cache['position_ids'] = torch.cat(
                            (cache['position_ids'], kwargs['position_ids']), dim=0
                        )
                raise ValueError

        self.layers[0] = self.layers[0].to(self.device)
        original_layer0 = self.layers[0]
        self.layers[0] = Catcher(self.layers[0])

        for batch in calib_loader:
            try:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                input_ids_list.append(batch['input_ids'].cpu())
                self.model(**batch)
            except ValueError:
                pass

        self.layers[0] = original_layer0
        self.layers[0] = self.layers[0].cpu()

        # Move embedding layers back to CPU
        if "opt" in self.model_name:
            self.model.model.decoder.embed_tokens = self.model.model.decoder.embed_tokens.cpu()
            self.model.model.decoder.embed_positions = self.model.model.decoder.embed_positions.cpu()
        else:
            self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()

        torch.cuda.empty_cache()

        attention_masks = cache['attention_mask']
        position_ids = cache.get('position_ids', None)

        # Process each layer with sensitivity estimation using proxy loss
        outs = torch.zeros_like(inps)

        for layer_idx in tqdm(range(len(self.layers))):
            layer = self.layers[layer_idx].to(self.device)
            subset = find_layers(layer)

            # First, compute original outputs (without SVD parameterization)
            original_outs = torch.zeros_like(inps)
            with torch.no_grad():
                for j in range(inps.shape[0]):
                    inp_j = inps[j].unsqueeze(0)
                    if position_ids is not None and "opt" not in self.model_name:
                        original_outs[j] = layer(inp_j,
                                                  attention_mask=attention_masks[j].unsqueeze(0),
                                                  position_ids=position_ids[j].unsqueeze(0))[0]
                    else:
                        original_outs[j] = layer(inp_j,
                                                  attention_mask=attention_masks[j].unsqueeze(0))[0]

            # Initialize Fisher accumulators for this layer
            layer_fisher = {}
            for name in subset:
                if layer_idx in self.svd_components and name in self.svd_components[layer_idx]:
                    U, S, VT, bias = self.svd_components[layer_idx][name]
                    layer_fisher[name] = torch.zeros_like(S)

            # Replace with SVD layers (sigma is differentiable)
            svd_layers = {}
            for name in subset:
                if layer_idx in self.svd_components and name in self.svd_components[layer_idx]:
                    U, S, VT, bias = self.svd_components[layer_idx][name]
                    svd_layer = SVDParameterizedLinear(
                        U.to(self.device),
                        S.to(self.device),
                        VT.to(self.device),
                        bias.to(self.device) if bias is not None else None
                    )
                    svd_layers[name] = svd_layer
                    self._set_module_by_name(layer, name, svd_layer)

            # Process each sample with proxy loss
            for j in range(inps.shape[0]):
                # Zero gradients
                for svd_layer in svd_layers.values():
                    if svd_layer.sigma.grad is not None:
                        svd_layer.sigma.grad.zero_()

                # Forward pass through current layer (with grad)
                inp_j = inps[j].unsqueeze(0)
                if position_ids is not None and "opt" not in self.model_name:
                    out_j = layer(inp_j,
                                  attention_mask=attention_masks[j].unsqueeze(0),
                                  position_ids=position_ids[j].unsqueeze(0))[0]
                else:
                    out_j = layer(inp_j,
                                  attention_mask=attention_masks[j].unsqueeze(0))[0]

                # Compute proxy loss: MSE between SVD output and original output
                # Plus variance term to encourage stability
                target = original_outs[j].unsqueeze(0).detach()
                diff = out_j.float() - target.float()
                loss_magnitude = (diff ** 2).mean()
                loss_variance = diff.var()
                loss = loss_magnitude + 0.1 * loss_variance

                # Backward pass
                loss.backward()

                # Accumulate squared gradients (Fisher information)
                for name, svd_layer in svd_layers.items():
                    if svd_layer.sigma.grad is not None:
                        layer_fisher[name] += svd_layer.sigma.grad.pow(2).cpu()

                # Store output for next layer
                with torch.no_grad():
                    outs[j] = out_j.detach()

            # Average Fisher information
            for name in layer_fisher:
                layer_fisher[name] /= inps.shape[0]

            self.fisher_info[layer_idx] = layer_fisher

            # Restore original linear layers for next iteration
            for name in subset:
                if layer_idx in self.svd_components and name in self.svd_components[layer_idx]:
                    U, S, VT, bias = self.svd_components[layer_idx][name]
                    W = torch.matmul(U * S, VT)
                    out_features, in_features = W.shape
                    new_linear = nn.Linear(in_features, out_features, bias=bias is not None)
                    new_linear.weight.data = W.to(subset[name].weight.dtype).to(self.device)
                    if bias is not None:
                        new_linear.bias.data = bias.to(subset[name].weight.dtype).to(self.device)
                    self._set_module_by_name(layer, name, new_linear)

            self.layers[layer_idx] = layer.cpu()
            inps = outs.clone()
            torch.cuda.empty_cache()

        print(f"  Estimated Fisher information for {len(self.layers)} layers")

    def _estimate_fisher_full(self, calib_loader: List[Dict]) -> None:
        """
        Full Fisher estimation using end-to-end backpropagation with true task loss.

        CORRECTED: Computes per-sample gradients for accurate Fisher information.

        Fisher information is defined as: F_ii = E[(∂L/∂σ_i)²]
        This requires computing gradients for EACH SAMPLE separately, then averaging
        the squared gradients. NOT squaring the average gradient.

        Memory-efficient implementation with:
        - Multi-GPU model parallelism (if available)
        - Gradient checkpointing
        - Per-sample gradient computation (correct Fisher)
        """

        print("  Using end-to-end task loss (cross-entropy) for Fisher estimation...")
        print("  Computing per-sample gradients for accurate Fisher information...")

        # Replace all layers with SVD-parameterized versions
        self._replace_with_svd_layers()

        # Distribute model across GPUs if multi-GPU is enabled
        if self.use_multi_gpu:
            print(f"  Distributing model across {len(self.devices)} GPUs...")
            self._distribute_model_across_gpus()
        else:
            self.model = self.model.to(self.device)

        self.model.train()

        # Enable gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("  Gradient checkpointing enabled")

        # Initialize Fisher accumulators using stored SVD layer references
        for layer_idx in self.svd_layer_refs:
            layer_fisher = {}
            for name, svd_layer in self.svd_layer_refs[layer_idx].items():
                layer_fisher[name] = torch.zeros_like(svd_layer.sigma.data, device='cpu')
            self.fisher_info[layer_idx] = layer_fisher

        # Accumulate Fisher information with PER-SAMPLE gradients
        num_samples = 0
        total_loss = 0.0
        target_device = self.devices[0] if self.use_multi_gpu else self.device

        for batch in tqdm(calib_loader):
            batch = {k: v.to(target_device) for k, v in batch.items()}
            batch_size = batch['input_ids'].shape[0]

            # Process each sample individually for correct Fisher estimation
            for sample_idx in range(batch_size):
                # Extract single sample
                single_sample = {k: v[sample_idx:sample_idx+1] for k, v in batch.items()}

                # Zero gradients
                self.model.zero_grad()

                try:
                    # Forward pass with cross-entropy loss for single sample
                    outputs = self.model(**single_sample, labels=single_sample['input_ids'])
                    loss = outputs.loss
                    total_loss += loss.item()

                    # Backward pass
                    loss.backward()

                    # Accumulate squared gradients (correct Fisher: E[grad²])
                    for layer_idx in self.svd_layer_refs:
                        for name, svd_layer in self.svd_layer_refs[layer_idx].items():
                            if svd_layer.sigma.grad is not None:
                                self.fisher_info[layer_idx][name] += svd_layer.sigma.grad.pow(2).cpu()

                    num_samples += 1

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"  Warning: OOM at sample {num_samples}, skipping...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

            # Clear cache after each batch
            torch.cuda.empty_cache()

        # Average Fisher information
        if num_samples > 0:
            for layer_idx in self.fisher_info:
                for name in self.fisher_info[layer_idx]:
                    self.fisher_info[layer_idx][name] /= num_samples

            avg_loss = total_loss / num_samples
            print(f"  Average calibration loss: {avg_loss:.4f}")
        else:
            print("  Warning: No samples processed successfully.")
            print("  Falling back to proxy loss estimation...")
            self._restore_original_layers()
            self._collect_model_to_cpu()
            if hasattr(self.model, 'gradient_checkpointing_disable'):
                self.model.gradient_checkpointing_disable()
            # Fall back to low resource mode
            self._estimate_fisher_low_resource(calib_loader)
            return

        # Disable gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()

        # Restore original model
        self._restore_original_layers()
        self._collect_model_to_cpu()
        self.model.eval()

        print(f"  Estimated Fisher information using {num_samples} samples (per-sample gradients)")

    def _distribute_model_across_gpus(self) -> None:
        """
        Distribute model layers across multiple GPUs for model parallelism.
        Adds forward hooks to automatically move tensors between devices.
        """
        num_layers = len(self.layers)
        layers_per_gpu = num_layers // len(self.devices)
        extra_layers = num_layers % len(self.devices)

        # Track which device each layer is on
        self.layer_devices = {}

        # Move embedding layers to first GPU
        if "opt" in self.model_name:
            self.model.model.decoder.embed_tokens = self.model.model.decoder.embed_tokens.to(self.devices[0])
            self.model.model.decoder.embed_positions = self.model.model.decoder.embed_positions.to(self.devices[0])
        else:
            self.model.model.embed_tokens = self.model.model.embed_tokens.to(self.devices[0])

        # Distribute transformer layers and track devices
        layer_idx = 0
        for gpu_idx, device in enumerate(self.devices):
            n_layers = layers_per_gpu + (1 if gpu_idx < extra_layers else 0)

            for _ in range(n_layers):
                if layer_idx < num_layers:
                    self.layers[layer_idx] = self.layers[layer_idx].to(device)
                    self.layer_devices[layer_idx] = device
                    layer_idx += 1

        # Move final norm and lm_head to last GPU
        last_device = self.devices[-1]
        if "opt" in self.model_name:
            self.model.model.decoder.final_layer_norm = self.model.model.decoder.final_layer_norm.to(last_device)
        else:
            self.model.model.norm = self.model.model.norm.to(last_device)

        if hasattr(self.model, 'lm_head'):
            self.model.lm_head = self.model.lm_head.to(last_device)

        # Add forward pre-hooks to move hidden states to correct device
        self.device_hooks = []

        def make_hook(target_device):
            def hook(module, args, kwargs):
                # Move all tensor args to target device
                new_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        new_args.append(arg.to(target_device))
                    else:
                        new_args.append(arg)
                # Move all tensor kwargs to target device
                new_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        new_kwargs[k] = v.to(target_device)
                    else:
                        new_kwargs[k] = v
                return tuple(new_args), new_kwargs
            return hook

        for idx in range(num_layers):
            device = self.layer_devices[idx]
            handle = self.layers[idx].register_forward_pre_hook(make_hook(device), with_kwargs=True)
            self.device_hooks.append(handle)

        # Add hook for final norm
        def norm_hook(module, args, kwargs):
            new_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    new_args.append(arg.to(last_device))
                else:
                    new_args.append(arg)
            new_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    new_kwargs[k] = v.to(last_device)
                else:
                    new_kwargs[k] = v
            return tuple(new_args), new_kwargs

        if "opt" in self.model_name:
            handle = self.model.model.decoder.final_layer_norm.register_forward_pre_hook(norm_hook, with_kwargs=True)
        else:
            handle = self.model.model.norm.register_forward_pre_hook(norm_hook, with_kwargs=True)
        self.device_hooks.append(handle)

        # Add hook for lm_head
        if hasattr(self.model, 'lm_head'):
            def lm_head_hook(module, args, kwargs):
                new_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        new_args.append(arg.to(last_device))
                    else:
                        new_args.append(arg)
                new_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        new_kwargs[k] = v.to(last_device)
                    else:
                        new_kwargs[k] = v
                return tuple(new_args), new_kwargs
            handle = self.model.lm_head.register_forward_pre_hook(lm_head_hook, with_kwargs=True)
            self.device_hooks.append(handle)

        print(f"  Model distributed: {layers_per_gpu}-{layers_per_gpu + 1} layers per GPU")
        print(f"  Added {len(self.device_hooks)} device transfer hooks")

    def _remove_device_hooks(self) -> None:
        """Remove all device transfer hooks."""
        if hasattr(self, 'device_hooks'):
            for handle in self.device_hooks:
                handle.remove()
            self.device_hooks = []

    def _collect_model_to_cpu(self) -> None:
        """
        Move all model components back to CPU.
        """
        # Remove hooks first
        self._remove_device_hooks()

        # Move embedding layers
        if "opt" in self.model_name:
            self.model.model.decoder.embed_tokens = self.model.model.decoder.embed_tokens.cpu()
            self.model.model.decoder.embed_positions = self.model.model.decoder.embed_positions.cpu()
            self.model.model.decoder.final_layer_norm = self.model.model.decoder.final_layer_norm.cpu()
        else:
            self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()
            self.model.model.norm = self.model.model.norm.cpu()

        # Move all layers
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx] = self.layers[layer_idx].cpu()

        # Move lm_head
        if hasattr(self.model, 'lm_head'):
            self.model.lm_head = self.model.lm_head.cpu()

        torch.cuda.empty_cache()

    def compute_importance_scores(self, fisher_lambda: float = 2.0) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute importance scores for all singular values using LOG-SPACE formula.

        Formula: Score_i = log(σ_i + ε) + λ × log(F_ii + ε)

        This is mathematically equivalent to: Score ∝ σ × F^λ (for ranking purposes)

        The log-space formulation has key advantages:
        1. Scale invariance: not affected by absolute magnitudes
        2. Balanced influence: compresses σ's huge range (0 to 70000) to ~11
        3. Numerical stability: avoids extreme values from σ² × F

        Theoretical justification:
        - When λ=1: equivalent to first-order Taylor approximation (ΔL ∝ σ × √F)
        - When λ=2: equivalent to second-order approximation (ΔL ∝ σ² × F)
        - λ>1 gives Fisher more influence to compensate for its smaller dynamic range

        Args:
            fisher_lambda: Weight for Fisher term in log space. Default=2.0
                          Higher values give Fisher more influence on ranking.

        Returns:
            Dictionary of importance scores per layer and sublayer
        """
        importance_scores = {}
        fisher_used = 0
        fisher_fallback = 0

        # Small epsilon to avoid log(0)
        eps = 1e-10

        # Collect statistics for diagnostics
        fisher_stats = {'min': float('inf'), 'max': 0, 'mean': 0, 'count': 0}
        sigma_stats = {'min': float('inf'), 'max': 0, 'mean': 0, 'count': 0}

        # For comparing old vs new formula rankings
        old_formula_scores = {}
        new_formula_scores = {}

        for layer_idx in self.svd_components:
            layer_scores = {}
            old_layer_scores = {}

            for name in self.svd_components[layer_idx]:
                U, S, VT, bias = self.svd_components[layer_idx][name]

                if layer_idx in self.fisher_info and name in self.fisher_info[layer_idx]:
                    F = self.fisher_info[layer_idx][name]

                    # Collect Fisher statistics
                    F_positive = F.clamp(min=eps)  # Ensure positive for log
                    fisher_stats['min'] = min(fisher_stats['min'], F_positive.min().item())
                    fisher_stats['max'] = max(fisher_stats['max'], F_positive.max().item())
                    fisher_stats['mean'] += F_positive.sum().item()
                    fisher_stats['count'] += len(F)

                    # Collect sigma statistics
                    S_positive = S.clamp(min=eps)  # Ensure positive for log
                    sigma_stats['min'] = min(sigma_stats['min'], S_positive.min().item())
                    sigma_stats['max'] = max(sigma_stats['max'], S_positive.max().item())
                    sigma_stats['mean'] += S_positive.sum().item()
                    sigma_stats['count'] += len(S)

                    # ============================================
                    # NEW: Log-space importance score
                    # Score = log(σ) + λ × log(F)
                    # ============================================
                    log_sigma = torch.log(S_positive)
                    log_fisher = torch.log(F_positive)
                    scores = log_sigma + fisher_lambda * log_fisher

                    # Also compute old formula for comparison
                    old_scores = S.pow(2) * F
                    old_layer_scores[name] = old_scores

                    fisher_used += 1
                else:
                    # Fallback to log magnitude-based scoring
                    S_positive = S.clamp(min=eps)
                    scores = torch.log(S_positive)  # Just log(σ)
                    old_layer_scores[name] = S.pow(2)
                    fisher_fallback += 1

                layer_scores[name] = scores

            importance_scores[layer_idx] = layer_scores
            old_formula_scores[layer_idx] = old_layer_scores

        # Print statistics
        print(f"  Importance scoring: LOG-SPACE formula")
        print(f"  Formula: Score = log(σ) + {fisher_lambda} × log(F)")
        print(f"  Projections: {fisher_used} with Fisher, {fisher_fallback} fallback to magnitude")

        if fisher_stats['count'] > 0:
            fisher_stats['mean'] /= fisher_stats['count']
            sigma_stats['mean'] /= sigma_stats['count']
            print(f"  Fisher stats: min={fisher_stats['min']:.2e}, max={fisher_stats['max']:.2e}, mean={fisher_stats['mean']:.2e}")
            print(f"  Sigma stats: min={sigma_stats['min']:.4f}, max={sigma_stats['max']:.4f}, mean={sigma_stats['mean']:.4f}")

            # Dynamic range in log space
            fisher_log_range = torch.log(torch.tensor(fisher_stats['max'])) - torch.log(torch.tensor(fisher_stats['min'] + eps))
            sigma_log_range = torch.log(torch.tensor(sigma_stats['max'])) - torch.log(torch.tensor(sigma_stats['min'] + eps))
            print(f"  Log-space ranges: log(σ) range={sigma_log_range.item():.1f}, log(F) range={fisher_log_range.item():.1f}")
            print(f"  Effective Fisher influence: {fisher_lambda} × {fisher_log_range.item():.1f} = {fisher_lambda * fisher_log_range.item():.1f}")

            # Compare rankings between old and new formula
            self._compare_ranking_formulas(old_formula_scores, importance_scores)

        return importance_scores

    def _compare_ranking_formulas(self, old_scores: Dict, new_scores: Dict) -> None:
        """Compare rankings between old (σ²×F) and new (log) formulas."""
        # Flatten all scores
        old_flat = []
        new_flat = []

        for layer_idx in old_scores:
            for name in old_scores[layer_idx]:
                old_s = old_scores[layer_idx][name]
                new_s = new_scores[layer_idx][name]
                for i in range(len(old_s)):
                    old_flat.append((old_s[i].item(), layer_idx, name, i))
                    new_flat.append((new_s[i].item(), layer_idx, name, i))

        # Sort by score (descending)
        old_flat.sort(key=lambda x: x[0], reverse=True)
        new_flat.sort(key=lambda x: x[0], reverse=True)

        # Compare top-K selections
        total = len(old_flat)
        for ratio in [0.2, 0.4, 0.6]:
            k = int(total * ratio)
            old_top_k = set((x[1], x[2], x[3]) for x in old_flat[:k])
            new_top_k = set((x[1], x[2], x[3]) for x in new_flat[:k])
            overlap = len(old_top_k & new_top_k)
            overlap_pct = overlap / k * 100
            print(f"  Top-{ratio:.0%} overlap (old vs new formula): {overlap_pct:.1f}%")

    def phase3_global_truncation(self, ratio: float, min_rank: int = 16,
                                   fisher_lambda: float = 2.0,
                                   use_residual_blocks: bool = False,
                                   block_share: float = 0.1) -> int:
        """
        Phase 3: Global truncation based on importance scores.

        Following the algorithm:
        1. Compute Score_i = log(σ_i) + λ × log(F_ii) for all singular values
        2. Apply layer position factor for balanced truncation
        3. Use ADAPTIVE min/max allocation:
           - f_min: Binary search to achieve target floor_share of budget
           - max_factor: Per-projection, based on score entropy/concentration
        4. Greedy allocation based on marginal utility
        5. Keep top-k singular values per projection (contiguous)

        Args:
            ratio: Target retention ratio (0-1). Higher means more parameters kept.
            min_rank: Minimum rank to keep per layer (default: 16)
            fisher_lambda: Weight for Fisher term in log-space formula (default: 2.0)
                          Higher values give Fisher more influence.
            use_residual_blocks: If True, reserve block_share of budget for blocks
            block_share: Fraction of budget to reserve for blocks (default: 0.1 = 10%)

        Returns:
            remaining_budget: Parameter budget remaining for residual blocks (0 if not used)
        """
        print(f"Phase 3: Global Truncation (target ratio: {ratio:.2%}, min_rank: {min_rank}, λ={fisher_lambda})...")
        if use_residual_blocks:
            print(f"  Reserving {block_share:.0%} of budget for residual blocks")

        # Compute importance scores using log-space formula
        importance_scores = self.compute_importance_scores(fisher_lambda=fisher_lambda)

        num_layers = len(self.layers)

        # Normalization strategy selection
        # Option 1: Per-layer normalization (current) - equalizes layers, may lose important signals
        # Option 2: No normalization - uses raw S² × F scores with layer_factor
        # Option 3: Global normalization - single normalization across all layers
        USE_NORMALIZATION = False  # Try without normalization to see if it helps

        normalized_scores = {}

        if USE_NORMALIZATION:
            # Per-LAYER normalization (preserves relative importance within layer)
            for layer_idx in importance_scores:
                normalized_scores[layer_idx] = {}

                # Collect all scores in this layer to compute layer-level norm
                all_layer_scores = []
                for name in importance_scores[layer_idx]:
                    all_layer_scores.append(importance_scores[layer_idx][name])

                # Concatenate and compute L2 norm across the entire layer
                all_scores_tensor = torch.cat(all_layer_scores)
                layer_norm = torch.norm(all_scores_tensor).item() + 1e-10

                # Layer position factor
                layer_position = layer_idx / (num_layers - 1) if num_layers > 1 else 0.5
                layer_factor = 0.5 + layer_position

                for name in importance_scores[layer_idx]:
                    scores = importance_scores[layer_idx][name]
                    normalized = scores / layer_norm * layer_factor
                    normalized_scores[layer_idx][name] = normalized

            print(f"  Using per-layer normalization with layer_factor")
        else:
            # NO normalization - use raw S² × F scores with layer_factor only
            # This preserves absolute importance information
            for layer_idx in importance_scores:
                normalized_scores[layer_idx] = {}

                # Layer position factor: later layers get higher weight
                layer_position = layer_idx / (num_layers - 1) if num_layers > 1 else 0.5
                layer_factor = 0.5 + layer_position  # Range: [0.5, 1.5]

                for name in importance_scores[layer_idx]:
                    scores = importance_scores[layer_idx][name]
                    # No normalization, just apply layer_factor to raw scores
                    normalized_scores[layer_idx][name] = scores * layer_factor

            print(f"  Using NO normalization (raw S² × F × layer_factor)")

        # Print layer factor info for debugging
        print(f"  Layer factors: L0={0.5:.2f}, L{num_layers//2}={0.5 + 0.5:.2f}, L{num_layers-1}={1.5:.2f}")

        # Collect all scores with their identifiers (layer_idx, name, singular_value_idx)
        # Use normalized scores for ranking but store original scores for debugging
        all_scores = []
        magnitude_scores = []  # For comparison: what would pure σ² ranking give?

        for layer_idx in normalized_scores:
            for name in normalized_scores[layer_idx]:
                scores = normalized_scores[layer_idx][name]
                original_scores = importance_scores[layer_idx][name]

                # Get singular values for magnitude comparison
                U, S, VT, bias = self.svd_components[layer_idx][name]

                for i, (norm_score, orig_score) in enumerate(zip(scores, original_scores)):
                    all_scores.append((norm_score.item(), layer_idx, name, i, orig_score.item()))
                    magnitude_scores.append((S[i].item() ** 2, layer_idx, name, i))

        # Sort by normalized importance (descending)
        all_scores.sort(key=lambda x: x[0], reverse=True)
        magnitude_scores.sort(key=lambda x: x[0], reverse=True)

        # Compare Fisher-based ranking vs magnitude-based ranking
        # How many of the top-K Fisher selections would also be in top-K magnitude?
        target_count = int(len(all_scores) * ratio * 0.8)  # Approximate target
        fisher_top_set = set((s[1], s[2], s[3]) for s in all_scores[:target_count])
        magnitude_top_set = set((s[1], s[2], s[3]) for s in magnitude_scores[:target_count])
        overlap = len(fisher_top_set & magnitude_top_set)
        overlap_pct = overlap / target_count * 100 if target_count > 0 else 0
        print(f"  Fisher vs Magnitude ranking overlap: {overlap_pct:.1f}% (100% = identical, 0% = completely different)")

        # Calculate total singular values and target count
        total_sv_count = len(all_scores)

        # ================================================================
        # GREEDY RANK ALLOCATION based on MARGINAL UTILITY
        # ================================================================
        # Core idea: "Each +1 rank costs (m+n) params. Which layer gives best ROI?"
        #
        # Key insight: We MUST keep top-k (contiguous), so we only compete
        # at each layer's "current frontier" (next unselected SV).
        #
        # Algorithm:
        # 1. Initialize all layers with k=0
        # 2. Priority queue with (priority, layer, current_k) where
        #    priority = Score[k] / Cost, Cost = m + n
        # 3. Greedy: pop best, allocate, push next candidate
        # 4. Stop when param budget exhausted
        # ================================================================

        import heapq

        # Step 1: Calculate total parameter budget
        total_original_params = 0
        projection_info = {}  # (layer_idx, name) -> (m, n, original_rank, scores)

        for layer_idx in self.svd_components:
            for name in self.svd_components[layer_idx]:
                U, S, VT, _ = self.svd_components[layer_idx][name]
                m, n = U.shape[0], VT.shape[1]
                original_rank = len(S)
                total_original_params += m * n

                # Get importance scores for this projection
                scores = importance_scores[layer_idx][name]

                # Apply layer factor to scores
                layer_position = layer_idx / (num_layers - 1) if num_layers > 1 else 0.5
                layer_factor = 0.5 + layer_position
                weighted_scores = scores * layer_factor

                # Pre-compute uniform rank for this projection
                uniform_rank = int(m * n * ratio / (m + n))
                uniform_rank = min(uniform_rank, original_rank)

                projection_info[(layer_idx, name)] = {
                    'm': m,
                    'n': n,
                    'cost': m + n,  # Cost per singular value
                    'original_rank': original_rank,
                    'scores': weighted_scores,
                    'layer_factor': layer_factor,
                    'uniform_rank': uniform_rank  # Pre-computed for binary search
                }

        # Target parameter budget
        total_target_params = int(total_original_params * ratio)

        # If using residual blocks, reserve a portion for them
        if use_residual_blocks:
            block_budget = int(total_target_params * block_share)
            target_params = total_target_params - block_budget
            print(f"  Total original params: {total_original_params:,}")
            print(f"  Total target params (ratio={ratio:.0%}): {total_target_params:,}")
            print(f"  SVD budget ({1-block_share:.0%}): {target_params:,}")
            print(f"  Block budget ({block_share:.0%}): {block_budget:,}")
        else:
            target_params = total_target_params
            block_budget = 0
            print(f"  Total original params: {total_original_params:,}")
            print(f"  Target params (ratio={ratio:.0%}): {target_params:,}")

        # ================================================================
        # ADAPTIVE MIN/MAX ALLOCATION (replacing fixed 0.3 and 1.5)
        # ================================================================
        #
        # Scheme A: Binary search to find f_min such that floor allocation
        #           consumes floor_share of target budget
        #
        # Scheme B: Per-projection max_factor based on score concentration
        #           (entropy-based: sharp distributions get higher max)
        # ================================================================

        import math

        # Step 2a: Compute per-projection entropy-based concentration (Scheme B)
        # This determines how "sharp" each projection's importance distribution is
        projection_concentration = {}

        for key, info in projection_info.items():
            scores = info['scores']

            # IMPORTANT: scores are in LOG-SPACE (log(σ) + λ*log(F))
            # They can be negative! Use softmax to convert to probability distribution
            # softmax(x_i) = exp(x_i) / Σexp(x_j)
            # This is equivalent to normalizing σ × F^λ
            scores_shifted = scores - scores.max()  # Numerical stability
            exp_scores = torch.exp(scores_shifted)
            p = exp_scores / exp_scores.sum()

            # Compute entropy: H = -Σ p_i log(p_i)
            # Use p.clamp(min=1e-20) to avoid log(0)
            entropy = -(p * torch.log(p.clamp(min=1e-20))).sum().item()

            # Normalize entropy: H_norm = H / log(n), range [0, 1]
            max_entropy = math.log(len(scores))
            entropy_norm = entropy / max_entropy if max_entropy > 0 else 0

            # Concentration = 1 - H_norm (1 = very sharp, 0 = very flat)
            concentration = 1.0 - entropy_norm
            projection_concentration[key] = concentration

            # Store for debugging
            info['concentration'] = concentration

        # Print concentration statistics
        conc_values = list(projection_concentration.values())
        print(f"  Score concentration: min={min(conc_values):.3f}, max={max(conc_values):.3f}, mean={sum(conc_values)/len(conc_values):.3f}")

        # Step 2b: Binary search to find optimal f_min (Scheme A)
        # Goal: floor allocation = floor_share × target_params
        # floor_share should be higher when compression is more aggressive
        floor_share = 0.5 + 0.2 * (1 - ratio)  # 0.5 at ratio=1, 0.7 at ratio=0
        floor_target = floor_share * target_params

        def compute_floor_params(f_min_candidate):
            """Compute total params if we use f_min_candidate as the min factor."""
            total = 0
            for key, info in projection_info.items():
                uniform_rank = info['uniform_rank']
                original_rank = info['original_rank']
                min_alloc = max(min_rank, int(uniform_rank * f_min_candidate))
                min_alloc = min(min_alloc, original_rank)
                total += min_alloc * info['cost']
            return total

        # Binary search for f_min in range [0.1, 0.9]
        f_min_lo, f_min_hi = 0.1, 0.9
        for _ in range(20):  # ~20 iterations for precision
            f_min_mid = (f_min_lo + f_min_hi) / 2
            floor_params = compute_floor_params(f_min_mid)
            if floor_params < floor_target:
                f_min_lo = f_min_mid
            else:
                f_min_hi = f_min_mid

        f_min_optimal = (f_min_lo + f_min_hi) / 2
        print(f"  Adaptive f_min: {f_min_optimal:.3f} (floor_share={floor_share:.2f})")

        # Step 2c: Calculate per-projection min and max ranks
        projection_min_rank = {}
        projection_max_rank = {}

        for key, info in projection_info.items():
            m, n = info['m'], info['n']
            original_rank = info['original_rank']
            uniform_rank = info['uniform_rank']
            concentration = projection_concentration[key]

            # MINIMUM: Use adaptive f_min from binary search
            min_alloc = max(min_rank, int(uniform_rank * f_min_optimal))
            min_alloc = min(min_alloc, original_rank)

            # MAXIMUM: Use concentration-based max_factor (Scheme B)
            # Sharp distribution (high concentration) → allow higher max
            # Flat distribution (low concentration) → still allow moderate flexibility
            #
            # Key insight from experiments:
            # - Old mapping: 1.1 + 0.9 * concentration → too conservative (mean=1.17)
            # - Best manual setting: fixed 1.5 → PPL=43.09
            # - New mapping: higher baseline (1.35) + smaller range
            #
            # This allows Fisher to redistribute ranks even when scores are relatively flat
            max_factor = 1.35 + 0.45 * concentration  # Range: [1.35, 1.80]
            max_alloc = min(original_rank, max(min_alloc, int(uniform_rank * max_factor)))

            projection_min_rank[key] = min_alloc
            projection_max_rank[key] = max_alloc

            # Store for debugging
            info['max_factor'] = max_factor

        # Print max_factor statistics
        max_factors = [info['max_factor'] for info in projection_info.values()]
        print(f"  Adaptive max_factor: min={min(max_factors):.2f}, max={max(max_factors):.2f}, mean={sum(max_factors)/len(max_factors):.2f}")

        # Diagnostic: Check if min_rank is constraining allocations
        constrained_count = 0
        for key, info in projection_info.items():
            if projection_min_rank[key] > info['uniform_rank']:
                constrained_count += 1

        if constrained_count > 0:
            print(f"  WARNING: min_rank={min_rank} is higher than theoretical uniform_rank for {constrained_count}/{len(projection_info)} projections")

        # Step 3: Pre-allocate MINIMUM ranks (mandatory)
        layer_allocated_rank = {}
        current_params = 0

        for key, info in projection_info.items():
            min_r = projection_min_rank[key]
            layer_allocated_rank[key] = min_r
            current_params += min_r * info['cost']

        print(f"  After min allocation: {current_params:,} params ({current_params/total_original_params*100:.1f}%)")

        # Check if minimum allocation already exceeds budget
        if current_params > target_params:
            print(f"  WARNING: Minimum allocation exceeds budget! Reducing proportionally...")
            scale = target_params / current_params * 0.95
            current_params = 0
            for key, info in projection_info.items():
                min_r = max(min_rank, int(projection_min_rank[key] * scale))
                layer_allocated_rank[key] = min_r
                projection_min_rank[key] = min_r  # Update min rank
                current_params += min_r * info['cost']
            print(f"  After scaling: {current_params:,} params")

        # Step 4: Build priority queue for remaining allocation
        # Use negative score for max-heap behavior (heapq is min-heap)
        # Priority = Score[k] / Cost (marginal utility per parameter)
        heap = []

        for key, info in projection_info.items():
            current_k = layer_allocated_rank[key]
            max_k = projection_max_rank[key]
            if current_k < max_k:
                # Next candidate is index current_k
                score = info['scores'][current_k].item()
                cost = info['cost']
                # Marginal utility = score / cost
                priority = score / cost
                # Push negative for max-heap behavior
                heapq.heappush(heap, (-priority, score, key[0], key[1], current_k))

        # Step 5: Greedy allocation (respecting max constraints)
        allocations_made = 0
        while heap and current_params < target_params:
            neg_priority, score, layer_idx, name, k = heapq.heappop(heap)
            key = (layer_idx, name)
            info = projection_info[key]
            max_k = projection_max_rank[key]

            # Check if this is still the current frontier
            if layer_allocated_rank[key] != k:
                # This entry is stale (already allocated), skip
                continue

            # Check if we've hit max for this projection
            if k >= max_k:
                continue

            # Check if adding this SV exceeds budget
            if current_params + info['cost'] > target_params:
                # Would exceed budget, but continue looking for cheaper options
                continue

            # Allocate this singular value
            layer_allocated_rank[key] = k + 1
            current_params += info['cost']
            allocations_made += 1

            # Push next candidate from this layer (if under max)
            next_k = k + 1
            if next_k < max_k:
                next_score = info['scores'][next_k].item()
                next_priority = next_score / info['cost']
                heapq.heappush(heap, (-next_priority, next_score, layer_idx, name, next_k))

        print(f"  Greedy allocations: {allocations_made}")
        print(f"  Final params: {current_params:,} ({current_params/total_original_params*100:.2f}%)")

        # Step 6: Build kept_indices (always contiguous: 0 to k-1)
        kept_indices: Dict[int, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
        kept_count = 0

        for key, k in layer_allocated_rank.items():
            layer_idx, name = key
            for i in range(k):
                kept_indices[layer_idx][name].add(i)
                kept_count += 1

        print(f"  Total singular values kept: {kept_count}")

        # Analyze allocation distribution
        layer_allocation = {}
        layer_target_rank = {}  # For comparison

        for layer_idx in self.svd_components:
            layer_total = 0
            layer_target = 0
            for name in self.svd_components[layer_idx]:
                key = (layer_idx, name)
                info = projection_info[key]
                m, n = info['m'], info['n']
                # What would uniform allocation give?
                uniform_rank = int(m * n * ratio / (m + n))
                uniform_rank = max(min_rank, min(uniform_rank, info['original_rank']))

                layer_total += layer_allocated_rank[key]
                layer_target += uniform_rank
                layer_target_rank[key] = uniform_rank

            layer_allocation[layer_idx] = (layer_total, layer_target)

        # Print allocation analysis
        under_target = sum(1 for l, (actual, target) in layer_allocation.items() if actual < target * 0.9)
        over_target = sum(1 for l, (actual, target) in layer_allocation.items() if actual > target * 1.1)
        at_target = len(layer_allocation) - under_target - over_target
        print(f"  Allocation: {under_target} layers <90%, {at_target} ~100%, {over_target} >110% of uniform")

        # Show extreme examples
        if layer_allocation:
            sorted_layers = sorted(layer_allocation.items(),
                                   key=lambda x: x[1][0]/x[1][1] if x[1][1] > 0 else 0)
            if len(sorted_layers) >= 2:
                min_layer, (min_actual, min_target) = sorted_layers[0]
                max_layer, (max_actual, max_target) = sorted_layers[-1]
                print(f"  Layer {min_layer}: {min_actual}/{min_target} ({min_actual/min_target*100:.0f}% of uniform)")
                print(f"  Layer {max_layer}: {max_actual}/{max_target} ({max_actual/max_target*100:.0f}% of uniform)")

        # Truncate SVD components
        truncation_samples = []
        rank_stats = []
        for layer_idx in self.svd_components:
            for name in self.svd_components[layer_idx]:
                U, S, VT, bias = self.svd_components[layer_idx][name]
                original_rank = len(S)

                # Get indices to keep, sorted by original order
                if layer_idx in kept_indices and name in kept_indices[layer_idx]:
                    indices = sorted(list(kept_indices[layer_idx][name]))
                else:
                    # Fallback: keep at least one singular value
                    indices = [0]

                if len(indices) == 0:
                    indices = [0]

                indices = torch.tensor(indices)
                new_rank = len(indices)
                rank_stats.append(new_rank)

                # Store sample for debug
                if len(truncation_samples) < 3:
                    truncation_samples.append(f"Layer {layer_idx} {name}: {original_rank} -> {new_rank}")

                # Truncate: keep only selected singular values
                U_trunc = U[:, indices]
                S_trunc = S[indices]
                VT_trunc = VT[indices, :]

                self.svd_components[layer_idx][name] = (U_trunc, S_trunc, VT_trunc, bias)

        # Analyze which singular value indices are kept
        # If Fisher works, it might keep non-contiguous indices (not just top-k)
        # If it keeps mostly contiguous top indices, Fisher isn't adding value
        contiguous_count = 0
        non_contiguous_count = 0
        total_projections = 0

        for layer_idx in kept_indices:
            for name in kept_indices[layer_idx]:
                indices_list = sorted(list(kept_indices[layer_idx][name]))
                if len(indices_list) > 0:
                    total_projections += 1
                    # Check if indices are contiguous from 0 (like pure top-k)
                    expected_contiguous = list(range(len(indices_list)))
                    if indices_list == expected_contiguous:
                        contiguous_count += 1
                    else:
                        non_contiguous_count += 1

        contiguous_pct = contiguous_count / total_projections * 100 if total_projections > 0 else 0
        print(f"  Selection pattern: {contiguous_pct:.1f}% contiguous (top-k), {100-contiguous_pct:.1f}% non-contiguous")
        print(f"    (100% contiguous = Fisher not helping, just keeping top singular values)")

        # Print truncation samples
        print("  Truncation examples:")
        for sample in truncation_samples:
            print(f"    {sample}")

        # Print rank statistics
        print(f"  Rank statistics: min={min(rank_stats)}, max={max(rank_stats)}, avg={sum(rank_stats)/len(rank_stats):.1f}")

        # Calculate actual compression ratio achieved
        kept_params = 0
        for layer_idx in self.svd_components:
            for name in self.svd_components[layer_idx]:
                U, S, VT, _ = self.svd_components[layer_idx][name]
                m = U.shape[0]
                n = VT.shape[1]
                r = len(S)
                kept_params += r * (m + n)

        actual_ratio = kept_params / total_original_params
        print(f"  Actual SVD compression ratio: {actual_ratio:.2%}")
        print(f"  Kept {kept_count} singular values out of {total_sv_count}")

        # Store for Phase 3b
        self._total_original_params = total_original_params
        self._svd_params_used = kept_params

        # Return remaining budget for blocks
        if use_residual_blocks:
            # Actual remaining = block_budget + (target_params - kept_params)
            actual_remaining = block_budget + max(0, target_params - kept_params)
            print(f"  Budget for residual blocks: {actual_remaining:,} params")
            return actual_remaining
        return 0

    def phase3b_residual_block_selection(self, block_budget: int = 0,
                                          block_size: int = 16,
                                          top_per_row: int = 8,
                                          use_fisher_weight: bool = True,
                                          layer_balance: str = "none") -> None:
        """
        Phase 3b: Select high-importance residual blocks to complement low-rank SVD.

        After truncation, W ≈ U_k @ S_k @ V_k^T has residual error.
        We select a small number of dense blocks from R = W - U_k S_k V_k^T
        that capture the most important remaining information.

        This allows using smaller rank k while compensating with critical residual blocks,
        effectively breaking the k(m+n) parameter constraint.

        NOTE: Budget is unified with Phase 3 - block_budget comes from Phase 3's
        reserved budget (block_share of total).

        Optimizations:
        - Uses heap with candidate cap to avoid memory explosion
        - Limits candidates per projection to 2x average budget share
        - Uses BOTH row and column Fisher importance for scoring

        Args:
            block_budget: Parameter budget for blocks (from Phase 3)
            block_size: Size of each block (default: 16)
            top_per_row: Max candidate blocks per row-tile to limit search (default: 8)
            use_fisher_weight: If False, only use Frobenius norm (for debugging)
            layer_balance: Layer balancing strategy:
                - "none": No layer bias, pure error*Fisher score (recommended)
                - "later": Favor later layers (0.5 + position)
                - "earlier": Favor earlier layers (1.5 - position)
        """
        import heapq

        params_per_block = block_size * block_size
        total_block_budget = block_budget // params_per_block

        print(f"Phase 3b: Residual Block Selection (budget={block_budget:,} params, {total_block_budget} blocks, block_size={block_size})...")
        print(f"  Fisher weighting: {'enabled' if use_fisher_weight else 'DISABLED (debug mode)'}")
        print(f"  Layer balance: {layer_balance}")

        # Initialize residual block storage
        self.residual_blocks = {}
        self.block_size = block_size
        num_layers = len(self.layers)

        if total_block_budget == 0:
            print("  No blocks to select (budget too small)")
            return

        # Count projections for per-projection candidate cap
        num_projections = sum(
            1 for layer_idx in self.svd_components
            for name in self.svd_components[layer_idx]
            if layer_idx in self.original_weights and name in self.original_weights[layer_idx]
        )

        # Estimate max candidates (for logging)
        # Each projection: n_row_tiles * top_per_row candidates
        sample_layer = next(iter(self.original_weights.keys()))
        sample_name = next(iter(self.original_weights[sample_layer].keys()))
        sample_shape = self.original_weights[sample_layer][sample_name].shape
        n_row_tiles = (sample_shape[0] + block_size - 1) // block_size
        max_cand_per_proj = n_row_tiles * top_per_row
        estimated_max_candidates = num_projections * max_cand_per_proj
        print(f"  Projections: {num_projections}, max candidates ~{estimated_max_candidates:,}")

        # Candidate cap per projection: 2x fair share (to allow some flexibility)
        cand_cap_per_proj = max(100, (total_block_budget * 2) // max(1, num_projections))

        # Use a min-heap to keep top-K globally
        # Heap stores (score, counter, block) - counter breaks ties to avoid dict comparison
        heap = []
        heap_counter = 0  # Unique counter to break ties
        total_candidates_seen = 0

        for layer_idx in self.svd_components:
            if layer_idx not in self.original_weights:
                continue

            # Layer factor based on layer_balance strategy
            layer_position = layer_idx / (num_layers - 1) if num_layers > 1 else 0.5
            if layer_balance == "later":
                layer_factor = 0.5 + layer_position  # L0=0.5, L31=1.5
            elif layer_balance == "earlier":
                layer_factor = 1.5 - layer_position  # L0=1.5, L31=0.5
            else:  # "none" - recommended
                layer_factor = 1.0  # No bias, pure error*Fisher score

            for name in self.svd_components[layer_idx]:
                if name not in self.original_weights[layer_idx]:
                    continue

                # Get original weight and SVD components
                W = self.original_weights[layer_idx][name]
                U, S, VT, bias = self.svd_components[layer_idx][name]

                # Move to GPU for computation
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                W_gpu = W.to(device)
                U_gpu = U.to(device)
                S_gpu = S.to(device)
                VT_gpu = VT.to(device)

                # Get importance weights from Fisher if available
                # Extract BOTH row (output) and column (input) importance
                row_importance = None
                col_importance = None
                if use_fisher_weight and hasattr(self, 'fisher_info') and layer_idx in self.fisher_info:
                    if name in self.fisher_info[layer_idx]:
                        fisher = self.fisher_info[layer_idx][name]
                        if len(fisher.shape) == 2:
                            # Fisher is [out, in], get both dimensions
                            row_importance = fisher.sum(dim=1).to(device)  # Sum over columns -> [out]
                            col_importance = fisher.sum(dim=0).to(device)  # Sum over rows -> [in]
                        else:
                            # 1D Fisher, assume it's row importance
                            row_importance = fisher.to(device)

                # Select candidate blocks for this projection with layer_factor
                blocks = select_residual_blocks(
                    W_gpu, U_gpu, S_gpu, VT_gpu,
                    block_size=block_size,
                    budget_blocks=cand_cap_per_proj,  # Limit candidates per projection
                    row_importance=row_importance,
                    col_importance=col_importance,
                    top_per_row=top_per_row,
                    layer_factor=layer_factor,
                    use_fisher_weight=use_fisher_weight
                )

                total_candidates_seen += len(blocks)

                # Add to global heap (use negative score for min-heap → max behavior)
                for blk in blocks:
                    blk["layer_idx"] = layer_idx
                    blk["name"] = name

                    if len(heap) < total_block_budget:
                        heapq.heappush(heap, (blk["score"], heap_counter, blk))
                        heap_counter += 1
                    elif blk["score"] > heap[0][0]:
                        heapq.heapreplace(heap, (blk["score"], heap_counter, blk))
                        heap_counter += 1

                # Clear GPU memory
                del W_gpu, U_gpu, S_gpu, VT_gpu
                if row_importance is not None:
                    del row_importance
                if col_importance is not None:
                    del col_importance
                torch.cuda.empty_cache()

        if len(heap) == 0:
            print("  No residual block candidates found")
            return

        # Extract selected blocks from heap (tuple is (score, counter, blk))
        selected = [blk for _, _, blk in heap]
        print(f"  Selected {len(selected)} blocks from {total_candidates_seen} candidates (heap-based)")

        # Warning if no filtering happened (budget >= candidates)
        if len(selected) == total_candidates_seen:
            print(f"  WARNING: All candidates selected (budget >= candidates). Consider reducing block_share.")

        # Organize by (layer_idx, name)
        for blk in selected:
            key = (blk["layer_idx"], blk["name"])
            if key not in self.residual_blocks:
                self.residual_blocks[key] = []
            self.residual_blocks[key].append({
                "row": blk["row"],
                "col": blk["col"],
                "row_end": blk.get("row_end", blk["row"] + block_size),
                "col_end": blk.get("col_end", blk["col"] + block_size),
                "val": blk["val"]
            })

        # Print statistics
        total_blocks_used = len(selected)
        total_block_params = total_blocks_used * params_per_block
        print(f"  Residual blocks: {total_blocks_used} blocks, {total_block_params:,} params")
        print(f"  Projections with blocks: {len(self.residual_blocks)}")

        # Show distribution across layers
        layer_block_counts = {}
        layer_score_sums = {}
        for blk in selected:
            layer_idx = blk["layer_idx"]
            if layer_idx not in layer_block_counts:
                layer_block_counts[layer_idx] = 0
                layer_score_sums[layer_idx] = 0.0
            layer_block_counts[layer_idx] += 1
            layer_score_sums[layer_idx] += blk["score"]

        if layer_block_counts:
            min_layer = min(layer_block_counts.items(), key=lambda x: x[1])
            max_layer = max(layer_block_counts.items(), key=lambda x: x[1])
            avg_blocks = sum(layer_block_counts.values()) / len(layer_block_counts)
            print(f"  Block distribution: min={min_layer[1]} (L{min_layer[0]}), max={max_layer[1]} (L{max_layer[0]}), avg={avg_blocks:.0f}")

            # Show score statistics by layer
            min_avg_score = min((layer_score_sums[l] / layer_block_counts[l], l) for l in layer_block_counts)
            max_avg_score = max((layer_score_sums[l] / layer_block_counts[l], l) for l in layer_block_counts)
            print(f"  Avg score by layer: min={min_avg_score[0]:.2e} (L{min_avg_score[1]}), max={max_avg_score[0]:.2e} (L{max_avg_score[1]})")

    def phase3b_omp_block_selection(self, calib_loader: List[Dict],
                                     block_budget: int = 0,
                                     block_size: int = 16,
                                     token_sample_ratio: float = 0.2,
                                     top_k_per_iter: int = 32) -> None:
        """
        OMP-style greedy block selection using activation-space metrics.

        Key improvements over standard selection:
        1. Activation-space scoring: ||X @ R_block^T||² instead of ||R_block||²
        2. Greedy selection with residual updating: select best blocks iteratively,
           update residual after each selection to avoid redundancy

        Algorithm (Orthogonal Matching Pursuit style):
        1. Compute activation-space residual: R = Y - Y_svd = X @ (W - W_svd)^T
        2. For each candidate block, compute reduction in residual: ||X_c @ B_opt^T||²
        3. Select top-k blocks with highest reduction
        4. Update residual to account for selected blocks
        5. Repeat until budget exhausted

        Args:
            calib_loader: Calibration data loader
            block_budget: Parameter budget for blocks
            block_size: Size of each block (default: 16)
            token_sample_ratio: Ratio of tokens to sample (default: 0.2)
            top_k_per_iter: Number of blocks to select per iteration (default: 32)
        """
        params_per_block = block_size * block_size
        total_block_budget = block_budget // params_per_block

        print(f"Phase 3b OMP: Greedy Block Selection (budget={block_budget:,} params, {total_block_budget} blocks)...")
        print(f"  Using activation-space metric: ||X @ R^T||²")
        print(f"  Greedy selection with top_k_per_iter={top_k_per_iter}")

        if total_block_budget == 0:
            print("  No blocks to select (budget too small)")
            return

        self.residual_blocks = {}
        self.block_size = block_size

        # Move embedding layers to device
        if "opt" in self.model_name:
            self.model.model.decoder.embed_tokens = self.model.model.decoder.embed_tokens.to(self.device)
            self.model.model.decoder.embed_positions = self.model.model.decoder.embed_positions.to(self.device)
        else:
            self.model.model.embed_tokens = self.model.model.embed_tokens.to(self.device)

        # Capture inputs to first layer
        dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros(
            (len(calib_loader), self.model.seqlen, self.model.config.hidden_size),
            dtype=dtype, device='cpu'
        )
        cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp[0].detach().cpu().to(inps.dtype)
                cache['i'] += 1
                if cache['attention_mask'] is None:
                    cache['attention_mask'] = kwargs['attention_mask'].cpu()
                    if 'position_ids' in kwargs:
                        cache['position_ids'] = kwargs['position_ids'].cpu()
                else:
                    cache['attention_mask'] = torch.cat(
                        (cache['attention_mask'], kwargs['attention_mask'].cpu()), dim=0
                    )
                    if 'position_ids' in kwargs:
                        cache['position_ids'] = torch.cat(
                            (cache['position_ids'], kwargs['position_ids'].cpu()), dim=0
                        )
                raise ValueError

        self.layers[0] = self.layers[0].to(self.device)
        original_layer0 = self.layers[0]
        self.layers[0] = Catcher(self.layers[0])

        for batch in calib_loader:
            try:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**batch)
            except ValueError:
                pass

        self.layers[0] = original_layer0
        self.layers[0] = self.layers[0].cpu()

        if "opt" in self.model_name:
            self.model.model.decoder.embed_tokens = self.model.model.decoder.embed_tokens.cpu()
            self.model.model.decoder.embed_positions = self.model.model.decoder.embed_positions.cpu()
        else:
            self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()

        torch.cuda.empty_cache()

        attention_masks = cache['attention_mask']
        position_ids = cache.get('position_ids', None)
        tokens_per_seq = max(1, int(self.model.seqlen * token_sample_ratio))

        b = block_size
        outs = torch.zeros_like(inps)

        # ==========================================
        # PASS 1: Compute total error per projection
        # ==========================================
        print("  Pass 1: Computing activation-space error per projection...")
        proj_total_errors = {}  # (layer_idx, name) -> total error
        proj_block_data = {}    # (layer_idx, name) -> (scores, R_blocks, m, n, n_row_tiles, n_col_tiles)

        for layer_idx in tqdm(range(len(self.layers)), desc="Pass 1 - Error Computation"):
            layer = self.layers[layer_idx].float().to(self.device)

            if layer_idx not in self.svd_components or layer_idx not in self.original_weights:
                # Forward through layer
                with torch.no_grad():
                    for j in range(inps.shape[0]):
                        inp_j = inps[j].unsqueeze(0).float().to(self.device)
                        mask_j = attention_masks[j].unsqueeze(0).to(self.device)
                        if position_ids is not None and "opt" not in self.model_name:
                            pos_j = position_ids[j].unsqueeze(0).to(self.device)
                            outs[j] = layer(inp_j, attention_mask=mask_j, position_ids=pos_j, use_cache=False)[0].cpu().to(dtype)
                        else:
                            outs[j] = layer(inp_j, attention_mask=mask_j, use_cache=False)[0].cpu().to(dtype)
                self.layers[layer_idx] = layer.to(dtype).cpu()
                inps = outs.clone()
                torch.cuda.empty_cache()
                continue

            subset = find_layers(layer)

            # Capture inputs to linear layers
            layer_inputs = {name: [] for name in subset}
            handles = []

            def make_hook(name):
                def hook(module, inp, out):
                    x = inp[0].detach().float()
                    if x.dim() == 2:
                        T = x.shape[0]
                        if T > tokens_per_seq:
                            idx = torch.randperm(T, device=x.device)[:tokens_per_seq]
                            x = x.index_select(0, idx)
                        layer_inputs[name].append(x.cpu())
                        return
                    if x.shape[1] > tokens_per_seq:
                        indices = torch.randperm(x.shape[1], device=x.device)[:tokens_per_seq]
                        x = x[:, indices, :]
                    layer_inputs[name].append(x.cpu())
                return hook

            for name in subset:
                handle = subset[name].register_forward_hook(make_hook(name))
                handles.append(handle)

            with torch.no_grad():
                for j in range(inps.shape[0]):
                    inp_j = inps[j].unsqueeze(0).float().to(self.device)
                    mask_j = attention_masks[j].unsqueeze(0).to(self.device)
                    if position_ids is not None and "opt" not in self.model_name:
                        pos_j = position_ids[j].unsqueeze(0).to(self.device)
                        _ = layer(inp_j, attention_mask=mask_j, position_ids=pos_j, use_cache=False)
                    else:
                        _ = layer(inp_j, attention_mask=mask_j, use_cache=False)

            for handle in handles:
                handle.remove()

            # Compute scores for each projection
            for name in subset:
                if name not in self.svd_components[layer_idx] or len(layer_inputs.get(name, [])) == 0:
                    continue
                if name not in self.original_weights[layer_idx]:
                    continue

                key = (layer_idx, name)

                # Get data
                U, S, VT, bias = self.svd_components[layer_idx][name]
                W_orig = self.original_weights[layer_idx][name].float().to(self.device)
                W_svd = ((U.to(self.device) * S.to(self.device)) @ VT.to(self.device)).float()

                X = torch.cat([x.reshape(-1, x.shape[-1]) for x in layer_inputs[name]], dim=0).to(self.device)
                del layer_inputs[name]

                m, n = W_orig.shape
                N = X.shape[0]

                # Token subsampling for speed (borrowed from reference)
                max_tokens = 8192
                if N > max_tokens:
                    indices = torch.randperm(N, device=X.device)[:max_tokens]
                    X = X[indices]
                    N = max_tokens

                n_row_tiles = (m + b - 1) // b
                n_col_tiles = (n + b - 1) // b

                R_weight = W_orig - W_svd

                # Pad and reshape for batch computation
                n_padded = n_col_tiles * b
                m_padded = n_row_tiles * b

                X_padded = torch.zeros(N, n_padded, device=self.device, dtype=X.dtype)
                X_padded[:, :n] = X

                R_padded = torch.zeros(m_padded, n_padded, device=self.device, dtype=R_weight.dtype)
                R_padded[:m, :n] = R_weight

                # X_blocks: [n_col_tiles, N, b] for chunked access
                X_blocks = X_padded.view(N, n_col_tiles, b).permute(1, 0, 2)
                R_blocks = R_padded.view(n_row_tiles, b, n_col_tiles, b).permute(0, 2, 1, 3)

                del X_padded, R_padded, X, W_orig, W_svd, R_weight
                torch.cuda.empty_cache()

                # Double-layer chunking for memory efficiency (borrowed from reference)
                scores = torch.zeros(n_row_tiles, n_col_tiles, device=self.device)
                col_chunk_size = min(32, n_col_tiles)
                row_chunk_size = min(64, n_row_tiles)

                for ci_start in range(0, n_col_tiles, col_chunk_size):
                    ci_end = min(ci_start + col_chunk_size, n_col_tiles)
                    X_chunk = X_blocks[ci_start:ci_end]  # [col_chunk, N, b]
                    R_col_chunk = R_blocks[:, ci_start:ci_end, :, :]  # [n_row, col_chunk, b, b]

                    for ri_start in range(0, n_row_tiles, row_chunk_size):
                        ri_end = min(ri_start + row_chunk_size, n_row_tiles)
                        R_sub = R_col_chunk[ri_start:ri_end]  # [row_chunk, col_chunk, b, b]

                        # einsum: 'cnk,rcjk->rcnj' then sum over n,j
                        contribution = torch.einsum('cnk,rcjk->rcnj', X_chunk, R_sub)
                        scores[ri_start:ri_end, ci_start:ci_end] = (contribution ** 2).sum(dim=(2, 3))
                        del contribution

                    del X_chunk, R_col_chunk
                    torch.cuda.empty_cache()

                # Store total error and block data
                total_error = scores.sum().item()
                proj_total_errors[key] = total_error
                proj_block_data[key] = (scores.cpu(), R_blocks.cpu(), m, n, n_row_tiles, n_col_tiles)

                del X_blocks, scores
                torch.cuda.empty_cache()

            # Forward through layer
            with torch.no_grad():
                for j in range(inps.shape[0]):
                    inp_j = inps[j].unsqueeze(0).float().to(self.device)
                    mask_j = attention_masks[j].unsqueeze(0).to(self.device)
                    if position_ids is not None and "opt" not in self.model_name:
                        pos_j = position_ids[j].unsqueeze(0).to(self.device)
                        outs[j] = layer(inp_j, attention_mask=mask_j, position_ids=pos_j, use_cache=False)[0].cpu().to(dtype)
                    else:
                        outs[j] = layer(inp_j, attention_mask=mask_j, use_cache=False)[0].cpu().to(dtype)

            self.layers[layer_idx] = layer.to(dtype).cpu()
            inps = outs.clone()
            torch.cuda.empty_cache()

        # ==========================================
        # PASS 2: Allocate budget and select blocks
        # ==========================================
        print("  Pass 2: Proportional budget allocation and block selection...")

        # Compute proportional budget allocation with log-smoothing
        # Problem: Linear allocation creates extreme skew (min=15, max=15274)
        # Solution: Use sqrt or log to smooth the distribution while preserving relative order
        total_error = sum(proj_total_errors.values())
        if total_error == 0:
            print("  No error to reduce, skipping block selection")
            return

        # Compute smoothed errors using sqrt (balances between uniform and pure proportional)
        import math
        smoothed_errors = {}
        for key, error in proj_total_errors.items():
            # sqrt smoothing: reduces extreme ratios while preserving relative order
            # error_ratio 1000:1 becomes sqrt_ratio ~31:1
            smoothed_errors[key] = math.sqrt(max(error, 1e-10))

        total_smoothed = sum(smoothed_errors.values())

        # Minimum budget per projection (ensure every projection gets some blocks)
        n_projections = len(proj_total_errors)
        min_budget_per_proj = max(1, total_block_budget // (n_projections * 10))  # At least 10% uniform
        reserved_budget = min_budget_per_proj * n_projections
        remaining_budget = total_block_budget - reserved_budget

        proj_budgets = {}
        for key, smoothed_err in smoothed_errors.items():
            # Base allocation (minimum) + proportional allocation (based on smoothed error)
            prop_budget = int(remaining_budget * smoothed_err / total_smoothed)
            proj_budgets[key] = min_budget_per_proj + prop_budget

        # Adjust to match total budget
        allocated = sum(proj_budgets.values())
        if allocated > total_block_budget:
            scale = total_block_budget / allocated
            for key in proj_budgets:
                proj_budgets[key] = max(1, int(proj_budgets[key] * scale))

        # Print allocation statistics
        layer_budgets = {}
        for (layer_idx, name), budget in proj_budgets.items():
            layer_budgets[layer_idx] = layer_budgets.get(layer_idx, 0) + budget
        if layer_budgets:
            min_lb = min(layer_budgets.items(), key=lambda x: x[1])
            max_lb = max(layer_budgets.items(), key=lambda x: x[1])
            print(f"  Budget allocation: min={min_lb[1]} (L{min_lb[0]}), max={max_lb[1]} (L{max_lb[0]}), ratio={max_lb[1]/max(min_lb[1],1):.1f}x")

        # Select blocks for each projection with score damping for diversity
        total_blocks_selected = 0
        for key, (scores, R_blocks, m, n, n_row_tiles, n_col_tiles) in proj_block_data.items():
            layer_idx, name = key
            budget = proj_budgets.get(key, 0)
            if budget == 0:
                continue

            # Cap at half of available blocks
            budget = min(budget, n_row_tiles * n_col_tiles // 2)

            # Use iterative selection with score damping (borrowed from reference)
            # This encourages spatial diversity - blocks in same row/col get dampened
            scores = scores.clone()  # Work on a copy
            selected_mask = torch.zeros(n_row_tiles, n_col_tiles, dtype=torch.bool)
            selected_blocks = []

            # Iterative selection with damping
            top_k_per_iter = min(128, budget)  # Batch size per iteration
            max_iters = (budget + top_k_per_iter - 1) // top_k_per_iter

            for _ in range(max_iters):
                if len(selected_blocks) >= budget:
                    break

                k = min(top_k_per_iter, budget - len(selected_blocks))
                flat_scores = scores.view(-1).clone()
                flat_scores[selected_mask.view(-1)] = -float('inf')

                valid_count = (flat_scores > 0).sum().item()
                if valid_count == 0:
                    break

                k = min(k, valid_count)
                top_scores, top_indices = torch.topk(flat_scores, k)

                for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
                    if score <= 0 or len(selected_blocks) >= budget:
                        break

                    ri = idx // n_col_tiles
                    ci = idx % n_col_tiles

                    if selected_mask[ri, ci]:
                        continue

                    row_start = ri * b
                    row_end = min(row_start + b, m)
                    col_start = ci * b
                    col_end = min(col_start + b, n)

                    B_val = R_blocks[ri, ci, :row_end-row_start, :col_end-col_start]

                    selected_mask[ri, ci] = True
                    selected_blocks.append({
                        'row': row_start,
                        'col': col_start,
                        'row_end': row_end,
                        'col_end': col_end,
                        'val': B_val,
                        'score': score
                    })

                    # Score damping for diversity (borrowed from reference)
                    # Blocks in same column/row will have reduced priority
                    scores[:, ci] *= 0.5  # Same column: 50% damping
                    scores[ri, :] *= 0.8  # Same row: 20% damping

            if selected_blocks:
                self.residual_blocks[key] = selected_blocks
                total_blocks_selected += len(selected_blocks)

        # Clean up
        del proj_block_data
        torch.cuda.empty_cache()

        # Print statistics
        total_block_params = total_blocks_selected * params_per_block
        print(f"  Total blocks selected: {total_blocks_selected}, params: {total_block_params:,}")
        print(f"  Projections with blocks: {len(self.residual_blocks)}")

        # Distribution statistics
        if self.residual_blocks:
            layer_counts = {}
            for (layer_idx, name), blocks in self.residual_blocks.items():
                layer_counts[layer_idx] = layer_counts.get(layer_idx, 0) + len(blocks)
            if layer_counts:
                min_l = min(layer_counts.items(), key=lambda x: x[1])
                max_l = max(layer_counts.items(), key=lambda x: x[1])
                print(f"  Block distribution: min={min_l[1]} (L{min_l[0]}), max={max_l[1]} (L{max_l[0]})")

    def phase3b_refine_blocks(self, calib_loader: List[Dict], token_sample_ratio: float = 0.2) -> None:
        """
        Refine residual block values using least squares on calibration data.

        Similar to ALS calibration for SVD, this optimizes block values to minimize
        reconstruction error on actual data rather than using static residual values.

        For each block at position (row, col):
            Original: B = R[row:row+b, col:col+b]  (static residual)
            Refined:  B = lstsq(X[:, col:col+b], R_target[:, row:row+b])
                      where R_target = Y - X @ W_svd^T

        Args:
            calib_loader: Calibration data loader
            token_sample_ratio: Ratio of tokens to sample per sequence (default: 0.2)
        """
        if not hasattr(self, 'residual_blocks') or len(self.residual_blocks) == 0:
            print("Phase 3b Refinement: No blocks to refine")
            return

        print(f"Phase 3b Refinement: Optimizing block values on calibration data...")

        # Move embedding layers to device
        if "opt" in self.model_name:
            self.model.model.decoder.embed_tokens = self.model.model.decoder.embed_tokens.to(self.device)
            self.model.model.decoder.embed_positions = self.model.model.decoder.embed_positions.to(self.device)
        else:
            self.model.model.embed_tokens = self.model.model.embed_tokens.to(self.device)

        # Capture inputs to first layer
        dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros(
            (len(calib_loader), self.model.seqlen, self.model.config.hidden_size),
            dtype=dtype, device='cpu'
        )
        cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp[0].detach().cpu().to(inps.dtype)
                cache['i'] += 1
                if cache['attention_mask'] is None:
                    cache['attention_mask'] = kwargs['attention_mask'].cpu()
                    if 'position_ids' in kwargs:
                        cache['position_ids'] = kwargs['position_ids'].cpu()
                else:
                    cache['attention_mask'] = torch.cat(
                        (cache['attention_mask'], kwargs['attention_mask'].cpu()), dim=0
                    )
                    if 'position_ids' in kwargs:
                        cache['position_ids'] = torch.cat(
                            (cache['position_ids'], kwargs['position_ids'].cpu()), dim=0
                        )
                raise ValueError

        self.layers[0] = self.layers[0].to(self.device)
        original_layer0 = self.layers[0]
        self.layers[0] = Catcher(self.layers[0])

        for batch in calib_loader:
            try:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**batch)
            except ValueError:
                pass

        self.layers[0] = original_layer0
        self.layers[0] = self.layers[0].cpu()

        # Move embedding layers back to CPU
        if "opt" in self.model_name:
            self.model.model.decoder.embed_tokens = self.model.model.decoder.embed_tokens.cpu()
            self.model.model.decoder.embed_positions = self.model.model.decoder.embed_positions.cpu()
        else:
            self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()

        torch.cuda.empty_cache()

        attention_masks = cache['attention_mask']
        position_ids = cache.get('position_ids', None)

        # Token sampling
        tokens_per_seq = max(1, int(self.model.seqlen * token_sample_ratio))

        # Process each layer
        outs = torch.zeros_like(inps)
        total_improvement = 0.0
        blocks_refined = 0

        for layer_idx in tqdm(range(len(self.layers))):
            layer = self.layers[layer_idx].float().to(self.device)

            # Check if this layer has blocks to refine
            layer_has_blocks = any(
                key[0] == layer_idx for key in self.residual_blocks.keys()
            )

            if not layer_has_blocks:
                # Just forward through this layer
                with torch.no_grad():
                    for j in range(inps.shape[0]):
                        inp_j = inps[j].unsqueeze(0).float().to(self.device)
                        mask_j = attention_masks[j].unsqueeze(0).to(self.device)
                        if position_ids is not None and "opt" not in self.model_name:
                            pos_j = position_ids[j].unsqueeze(0).to(self.device)
                            outs[j] = layer(inp_j, attention_mask=mask_j, position_ids=pos_j, use_cache=False)[0].cpu().to(dtype)
                        else:
                            outs[j] = layer(inp_j, attention_mask=mask_j, use_cache=False)[0].cpu().to(dtype)
                self.layers[layer_idx] = layer.to(dtype).cpu()
                inps = outs.clone()
                torch.cuda.empty_cache()
                continue

            subset = find_layers(layer)

            # Capture inputs to each linear layer
            layer_inputs = {name: [] for name in subset}
            handles = []

            def make_hook(name):
                def hook(module, inp, out):
                    x = inp[0].detach().float()
                    if x.dim() == 2:
                        T = x.shape[0]
                        if T > tokens_per_seq:
                            idx = torch.randperm(T, device=x.device)[:tokens_per_seq]
                            x = x.index_select(0, idx)
                        layer_inputs[name].append(x.cpu())
                        return
                    if x.shape[1] > tokens_per_seq:
                        indices = torch.randperm(x.shape[1], device=x.device)[:tokens_per_seq]
                        x = x[:, indices, :]
                    layer_inputs[name].append(x.cpu())
                return hook

            for name in subset:
                handle = subset[name].register_forward_hook(make_hook(name))
                handles.append(handle)

            # Forward pass to capture inputs
            with torch.no_grad():
                for j in range(inps.shape[0]):
                    inp_j = inps[j].unsqueeze(0).float().to(self.device)
                    mask_j = attention_masks[j].unsqueeze(0).to(self.device)
                    if position_ids is not None and "opt" not in self.model_name:
                        pos_j = position_ids[j].unsqueeze(0).to(self.device)
                        _ = layer(inp_j, attention_mask=mask_j, position_ids=pos_j, use_cache=False)
                    else:
                        _ = layer(inp_j, attention_mask=mask_j, use_cache=False)

            for handle in handles:
                handle.remove()

            # Refine blocks for each projection
            for name in subset:
                key = (layer_idx, name)
                if key not in self.residual_blocks or len(layer_inputs.get(name, [])) == 0:
                    continue

                blocks = self.residual_blocks[key]
                if len(blocks) == 0:
                    continue

                # Get SVD components and TRUE original weight (before any SVD/calibration)
                U, S, VT, bias = self.svd_components[layer_idx][name]

                # Stack calibration inputs
                X = torch.cat([x.reshape(-1, x.shape[-1]) for x in layer_inputs[name]], dim=0).to(self.device)
                del layer_inputs[name]

                # IMPORTANT: Use stored original weights, NOT the linear layer's current weight
                # (which has been updated by ALS to be U @ S @ VT)
                if hasattr(self, 'original_weights') and layer_idx in self.original_weights and name in self.original_weights[layer_idx]:
                    W_orig = self.original_weights[layer_idx][name].float().to(self.device)
                else:
                    # Fallback: skip refinement for this projection (can't compute proper residual)
                    print(f"    Warning: original_weights not available for L{layer_idx} {name}, skipping refinement")
                    continue

                # Target output from TRUE original weight
                Y = X @ W_orig.T

                # SVD approximation output (from calibrated SVD)
                W_svd = ((U.to(self.device) * S.to(self.device)) @ VT.to(self.device)).float()
                Y_svd = X @ W_svd.T

                # Residual target - will be updated as we refine blocks
                R_target = Y - Y_svd  # [N, out_features]

                block_size = self.block_size

                # Sort blocks by row to process same-row blocks together
                # and update residual properly
                blocks_sorted = sorted(blocks, key=lambda b: (b["row"], b["col"]))

                # Refine each block with residual updating
                for blk in blocks_sorted:
                    row, col = blk["row"], blk["col"]
                    row_end = blk.get("row_end", row + block_size)
                    col_end = blk.get("col_end", col + block_size)

                    # Input slice for this block
                    X_c = X[:, col:col_end]  # [N, b]

                    # Target residual for this block's output positions
                    # NOTE: R_target is updated after each block refinement
                    R_c = R_target[:, row:row_end]  # [N, b]

                    # Least squares: find B such that X_c @ B^T ≈ R_c
                    try:
                        B_T = torch.linalg.lstsq(X_c, R_c).solution  # [col_size, row_size]
                        B_refined = B_T.T  # [row_size, col_size]

                        # Check for numerical issues
                        if torch.isnan(B_refined).any() or torch.isinf(B_refined).any():
                            continue

                        # Compute improvement
                        old_val = blk["val"].to(self.device)
                        loss_before = ((X_c @ old_val.T - R_c) ** 2).mean().item()
                        loss_after = ((X_c @ B_refined.T - R_c) ** 2).mean().item()

                        # Only use refined value if it's better
                        if loss_after < loss_before:
                            blk["val"] = B_refined.cpu()
                            if loss_before > 1e-10:
                                improvement = (1 - loss_after / loss_before) * 100
                                total_improvement += improvement
                            blocks_refined += 1

                            # KEY FIX: Update residual to account for this block's contribution
                            # This ensures subsequent blocks at same row see the remaining residual
                            R_target[:, row:row_end] = R_target[:, row:row_end] - X_c @ B_refined.T
                    except Exception:
                        continue

                del X, Y, Y_svd, R_target, W_orig, W_svd
                torch.cuda.empty_cache()

            # Forward through layer for next layer's input
            with torch.no_grad():
                for j in range(inps.shape[0]):
                    inp_j = inps[j].unsqueeze(0).float().to(self.device)
                    mask_j = attention_masks[j].unsqueeze(0).to(self.device)
                    if position_ids is not None and "opt" not in self.model_name:
                        pos_j = position_ids[j].unsqueeze(0).to(self.device)
                        outs[j] = layer(inp_j, attention_mask=mask_j, position_ids=pos_j, use_cache=False)[0].cpu().to(dtype)
                    else:
                        outs[j] = layer(inp_j, attention_mask=mask_j, use_cache=False)[0].cpu().to(dtype)

            self.layers[layer_idx] = layer.to(dtype).cpu()
            inps = outs.clone()
            torch.cuda.empty_cache()

        avg_improvement = total_improvement / blocks_refined if blocks_refined > 0 else 0
        print(f"  Refined {blocks_refined} blocks, avg improvement: {avg_improvement:.1f}%")

    def apply_compression(self, ratio: float, use_residual_blocks: bool = True) -> None:
        """
        Apply compression to the model by replacing layers with SVD-factorized versions.

        Note: Since global truncation produces different ranks per layer/projection,
        we need to create SVD modules with the actual truncated ranks, not a uniform ratio.

        Args:
            ratio: Compression ratio (0-1) - used only for module creation reference
            use_residual_blocks: Whether to use residual blocks if available (default: True)
        """
        print("Applying compression to model...")

        # Check if residual blocks are available
        has_residual_blocks = hasattr(self, 'residual_blocks') and len(self.residual_blocks) > 0
        if use_residual_blocks and has_residual_blocks:
            print(f"  Using residual blocks: {len(self.residual_blocks)} projections")
            block_size = getattr(self, 'block_size', 16)
        else:
            has_residual_blocks = False

        # First, compute actual ranks for each layer to determine per-layer ratios
        layer_ranks = {}
        total_original_params = 0
        total_compressed_params = 0
        total_block_params = 0

        for layer_idx in self.svd_components:
            layer_ranks[layer_idx] = {}
            for name in self.svd_components[layer_idx]:
                U, S, VT, bias = self.svd_components[layer_idx][name]
                actual_rank = len(S)
                layer_ranks[layer_idx][name] = actual_rank
                m, n = U.shape[0], VT.shape[1]
                total_original_params += m * n
                total_compressed_params += actual_rank * (m + n)

                # Count residual block params
                if has_residual_blocks and (layer_idx, name) in self.residual_blocks:
                    num_blocks = len(self.residual_blocks[(layer_idx, name)])
                    total_block_params += num_blocks * block_size * block_size

        # Print compression summary
        total_params = total_compressed_params + total_block_params
        print(f"  Original params: {total_original_params:,}")
        print(f"  SVD params: {total_compressed_params:,}")
        if total_block_params > 0:
            print(f"  Block params: {total_block_params:,}")
            print(f"  Total compressed: {total_params:,}")
        print(f"  Compression ratio: {total_params / total_original_params:.2%}")

        replaced_count = 0
        for layer_idx in tqdm(range(len(self.layers))):
            layer = self.layers[layer_idx]
            subset = find_layers(layer)

            if layer_idx not in self.svd_components:
                continue

            dtype = next(iter(self.model.parameters())).dtype

            # For each linear layer, create properly sized SVD factorization
            for name in subset:
                if name not in self.svd_components[layer_idx]:
                    continue

                U, S, VT, bias = self.svd_components[layer_idx][name]
                actual_rank = len(S)
                original_rank = min(U.shape[0], VT.shape[1])

                if actual_rank == 0:
                    continue

                # Check for NaN/Inf in SVD components - skip if corrupted
                if torch.isnan(U).any() or torch.isnan(S).any() or torch.isnan(VT).any():
                    print(f"  Warning: Layer {layer_idx} {name} has NaN in SVD components, skipping")
                    continue
                if torch.isinf(U).any() or torch.isinf(S).any() or torch.isinf(VT).any():
                    print(f"  Warning: Layer {layer_idx} {name} has Inf in SVD components, skipping")
                    continue
                # Ensure S is positive (required for sqrt)
                if (S < 0).any():
                    print(f"  Warning: Layer {layer_idx} {name} has negative S values, clamping")
                    S = torch.clamp(S, min=1e-8)

                # Debug: print first layer's compression
                if layer_idx == 0 and replaced_count < 2:
                    print(f"  Layer {layer_idx} {name}: rank {original_rank} -> {actual_rank}")

                # Compute U' = U @ sqrt(Sigma) and V' = sqrt(Sigma) @ VT
                sqrt_sigma = torch.sqrt(S)
                # U' shape: (out_features, rank), V' shape: (rank, in_features)
                svd_u = (U * sqrt_sigma).to(dtype)  # Broadcasting: U * sqrt_sigma
                svd_v = (sqrt_sigma.unsqueeze(1) * VT).to(dtype)  # sqrt_sigma @ VT

                out_features, in_features = U.shape[0], VT.shape[1]

                # Create new linear layers with correct sizes
                u_proj = nn.Linear(actual_rank, out_features, bias=(bias is not None))
                v_proj = nn.Linear(in_features, actual_rank, bias=False)

                u_proj.weight.data = svd_u
                v_proj.weight.data = svd_v
                if bias is not None:
                    u_proj.bias.data = bias.to(dtype)

                # Get residual blocks for this projection if available
                blocks = None
                if has_residual_blocks and (layer_idx, name) in self.residual_blocks:
                    blocks = self.residual_blocks[(layer_idx, name)]

                # Replace in model using a wrapper or direct replacement
                self._replace_linear_with_svd(layer, name, u_proj, v_proj, layer_idx, blocks, block_size if has_residual_blocks else 16)
                replaced_count += 1

            torch.cuda.empty_cache()

        print(f"  Replaced {replaced_count} linear layers with SVD factorization")

    def _replace_linear_with_svd(self, layer, name: str, u_proj: nn.Linear,
                                  v_proj: nn.Linear, layer_idx: int,
                                  blocks: Optional[List[Dict]] = None,
                                  block_size: int = 16) -> None:
        """
        Replace a linear layer with SVD factorization (V @ U), optionally with residual blocks.

        Args:
            layer: The transformer layer
            name: Name of the linear layer (e.g., "self_attn.q_proj")
            u_proj: The U projection (rank -> out_features)
            v_proj: The V projection (in_features -> rank)
            layer_idx: Layer index for model-specific handling
            blocks: Optional list of residual blocks
            block_size: Block size for residual blocks
        """
        # Create appropriate SVD layer based on whether blocks are available
        if blocks is not None and len(blocks) > 0:
            # Pack blocks into efficient format
            # IMPORTANT: Convert blocks to model dtype to avoid dtype mismatch in matmul
            device = u_proj.weight.device
            dtype = u_proj.weight.dtype
            groups = SVDLinearWithDenseBlocks.pack_blocks_by_col(blocks, block_size, device, dtype)
            svd_linear = SVDLinearWithDenseBlocks(v_proj, u_proj, block_size, groups)
        else:
            svd_linear = SVDLinear(v_proj, u_proj)

        # Navigate to the correct location and replace
        parts = name.split('.')
        module = layer
        for part in parts[:-1]:
            module = getattr(module, part)
        setattr(module, parts[-1], svd_linear)

    def compress(self, calib_loader: List[Dict], ratio: float,
                 whitening_mat: Optional[Dict] = None,
                 use_low_resource: bool = False,
                 calibration_steps: int = 50,
                 min_rank: int = 16,
                 fisher_lambda: float = 2.0,
                 use_als: bool = True,
                 als_iters: int = 2,
                 token_sample_ratio: float = 0.2,
                 use_fisher_weight_als: bool = False,
                 use_residual_blocks: bool = False,
                 block_share: float = 0.1,
                 block_size: int = 16,
                 use_block_fisher_weight: bool = True,
                 block_layer_balance: str = "none",
                 refine_blocks: bool = True,
                 use_omp_selection: bool = False,
                 omp_top_k_per_iter: int = 32,
                 joint_optimize_iters: int = 0,
                 use_distillation: bool = False,
                 distill_steps: int = 2000,
                 distill_lr: float = 1e-4,
                 distill_temperature: float = 2.0,
                 teacher_model_path: Optional[str] = None,
                 use_8bit_teacher: bool = True,
                 offline_logits_path: Optional[str] = None,
                 distill_checkpoint_dir: Optional[str] = None,
                 distill_gradient_accumulation: int = 4) -> nn.Module:
        """
        Full compression pipeline.

        Args:
            calib_loader: Calibration data loader
            ratio: Target compression ratio (0-1)
            whitening_mat: Optional whitening matrices from SVD-LLM
            use_low_resource: Use memory-efficient proxy loss (default: False, use true CE loss)
            calibration_steps: Number of calibration steps per layer (default: 50)
            min_rank: Minimum rank to keep per projection (default: 16)
            fisher_lambda: Weight for Fisher in log-space formula (default: 2.0)
                          Formula: Score = log(σ) + λ × log(F)
            use_als: Use ALS calibration instead of M-optimization (default: True)
            als_iters: Number of ALS iterations per layer (default: 2)
            token_sample_ratio: Ratio of tokens to sample per sequence for ALS (default: 0.1)
            use_fisher_weight_als: Use Fisher-weighted ALS calibration (default: False)
                                  Weights output dimensions by their sensitivity to task loss.
                                  F_y ≈ (U² @ F_σ) approximates output Fisher from singular value Fisher.
            use_residual_blocks: Use dense residual blocks to improve accuracy (default: False)
            block_share: Fraction of total budget for residual blocks (default: 0.1 = 10%)
                        Budget is unified: SVD gets (1-block_share), blocks get block_share
            block_size: Size of residual blocks (default: 16)
            use_block_fisher_weight: Use Fisher weighting for block selection (default: True)
                                    Set False for debugging (use pure Frobenius norm)
            block_layer_balance: Layer balance strategy for block selection (default: "none")
                                - "none": No layer bias, pure error*Fisher score (recommended)
                                - "later": Favor later layers
                                - "earlier": Favor earlier layers
            refine_blocks: Refine block values using lstsq on calibration data (default: True)
                          Similar to ALS for SVD, this optimizes block values to minimize
                          reconstruction error instead of using static residual values.
            use_omp_selection: Use OMP-style greedy block selection (default: False)
                              When True, uses activation-space metrics ||X @ R^T||² and
                              greedy selection with residual updating. More accurate but slower.
            omp_top_k_per_iter: Number of blocks to select per OMP iteration (default: 32)
            joint_optimize_iters: Number of joint optimization iterations after adding blocks (default: 0)
                                 Each iteration: ALS(SVD) -> Refine(Blocks) -> repeat
                                 This helps SVD and blocks jointly minimize reconstruction error.
            use_distillation: Enable Phase 5 distillation fine-tuning (default: False)
            distill_steps: Number of distillation training steps (default: 2000)
            distill_lr: Learning rate for distillation (default: 1e-4)
            distill_temperature: Distillation temperature (default: 2.0)
            teacher_model_path: Path to teacher model (default: None, uses model_name)
            use_8bit_teacher: Load teacher in 8-bit to save memory (default: True)
            offline_logits_path: Path to pre-computed teacher logits (default: None)
            distill_checkpoint_dir: Directory to save distillation checkpoints (default: None)
            distill_gradient_accumulation: Gradient accumulation steps for distillation (default: 4)

        Returns:
            Compressed model
        """
        # Phase 1: SVD Decomposition
        # Store original weights if we'll need them for residual block selection
        self.phase1_svd_decomposition(whitening_mat, store_original_weights=use_residual_blocks)

        # Phase 2: Sensitivity Estimation
        self.phase2_sensitivity_estimation(calib_loader, use_low_resource)

        # Phase 3: Global Truncation with adaptive min/max allocation
        # If using residual blocks, budget is unified - Phase 3 reserves block_share for blocks
        block_budget = self.phase3_global_truncation(
            ratio, min_rank=min_rank, fisher_lambda=fisher_lambda,
            use_residual_blocks=use_residual_blocks, block_share=block_share
        )

        # Phase 4: Layer-wise Calibration (optimize SVD factors to minimize reconstruction error)
        # NOTE: Run BEFORE Phase 3b so blocks capture post-ALS residual
        if calibration_steps > 0:
            try:
                if use_als:
                    fisher_str = " (Fisher-weighted)" if use_fisher_weight_als else ""
                    print(f"Starting Phase 4 ALS calibration ({als_iters} iterations){fisher_str}...")
                    self.phase4_als_calibration(calib_loader, num_iters=als_iters,
                                                 update_sigma=True, token_sample_ratio=token_sample_ratio,
                                                 use_fisher_weight=use_fisher_weight_als)
                else:
                    print(f"Starting Phase 4 M-optimization calibration...")
            except Exception as e:
                print(f"  Warning: Phase 4 calibration failed ({type(e).__name__}: {e}), skipping...")
                print("  Proceeding without calibration.")
                import traceback
                traceback.print_exc()
        else:
            print("Phase 4: Skipped (calibration_steps=0)")

        # Phase 3b: Residual Block Selection (if enabled)
        # NOTE: Run AFTER Phase 4 so blocks capture post-calibration residual
        if use_residual_blocks and block_budget > 0:
            self.block_size = block_size  # Store for refinement

            if use_omp_selection:
                # OMP-style greedy selection with activation-space metrics
                self.phase3b_omp_block_selection(
                    calib_loader=calib_loader,
                    block_budget=block_budget,
                    block_size=block_size,
                    token_sample_ratio=token_sample_ratio,
                    top_k_per_iter=omp_top_k_per_iter
                )

                # Refine block values using lstsq on calibration data
                # OMP selects blocks based on activation-space error but uses static residual values
                # Refinement optimizes these values to minimize actual reconstruction error
                if refine_blocks and len(self.residual_blocks) > 0:
                    self.phase3b_refine_blocks(calib_loader, token_sample_ratio=token_sample_ratio)
            else:
                # Standard selection based on weight-space residual
                self.phase3b_residual_block_selection(
                    block_budget=block_budget,
                    block_size=block_size,
                    top_per_row=8,
                    use_fisher_weight=use_block_fisher_weight,
                    layer_balance=block_layer_balance
                )

                # Phase 3b Refinement: Optimize block values using least squares
                if refine_blocks and len(self.residual_blocks) > 0:
                    self.phase3b_refine_blocks(calib_loader, token_sample_ratio=token_sample_ratio)

            # Joint optimization: alternate between ALS and block refinement
            # This helps SVD and blocks jointly minimize reconstruction error
            if joint_optimize_iters > 0 and len(self.residual_blocks) > 0:
                print(f"Phase 4b: Joint optimization ({joint_optimize_iters} iterations)...")
                for joint_iter in range(joint_optimize_iters):
                    print(f"  Joint iteration {joint_iter + 1}/{joint_optimize_iters}")

                    # Re-run ALS to adjust SVD components (now accounting for blocks)
                    # The blocks capture some error, so SVD should shift to capture different patterns
                    self.phase4_als_calibration_with_blocks(
                        calib_loader,
                        num_iters=1,
                        token_sample_ratio=token_sample_ratio
                    )

                    # Re-run block refinement with updated SVD
                    if refine_blocks:
                        self.phase3b_refine_blocks(calib_loader, token_sample_ratio=token_sample_ratio)

            # Clean up original weights to save memory
            if hasattr(self, 'original_weights'):
                del self.original_weights
                torch.cuda.empty_cache()

        # Apply compression to model
        self.apply_compression(ratio)

        return self.model


    def _get_module_by_name(self, parent: nn.Module, name: str) -> nn.Module:
        """Get a submodule by its name path."""
        parts = name.split('.')
        module = parent
        for part in parts:
            module = getattr(module, part)
        return module

    def phase4_als_calibration(self, calib_loader: List[Dict], num_iters: int = 2,
                                update_sigma: bool = True, token_sample_ratio: float = 0.2,
                                use_fisher_weight: bool = False) -> None:
        """
        Phase 4: ALS (Alternating Least Squares) Calibration.

        Unlike the M-optimization approach, ALS iteratively updates U and V separately:
        - Step A: Fix V, Σ, solve for U using least squares
        - Step B: Fix U, Σ, solve for V using least squares (FIXED: no U orthogonality assumption)
        - Step C (optional): Fix U, V, solve for diagonal scaling D (FIXED: r×r linear system)

        After each projection is calibrated, we write the updated SVD components back to
        the layer so that subsequent layers see the calibrated outputs.

        Fisher-Weighted ALS (use_fisher_weight=True):
        Standard ALS minimizes: ||X @ W^T - Y||²_F (all outputs equally weighted)
        Fisher-weighted minimizes: ||F^{1/2} @ (X @ W^T - Y)^T||²_F
        where F_y = diag(F_11, ..., F_mm) weights outputs by their sensitivity to loss.

        Output Fisher F_y is approximated from singular value Fisher F_σ:
            F_y ≈ diag(U @ diag(F_σ) @ U^T) = (U² @ F_σ)

        Mathematical formulation:
        Given W' = U @ Σ @ V^T, we want to minimize ||X @ W'^T - X @ W^T||_F^2

        Step A: Fix V, Σ, solve U
            Z = X @ V @ Σ (N × r)
            U^T = (Z^T Z)^{-1} Z^T Y → U = (solution)^T
            (Fisher does not affect Step A - it cancels out)

        Step B: Fix U, Σ, solve V (CORRECTED - no U orthogonality assumption)
            Without Fisher: Z = Y @ U_s @ (U_s^T U_s)^{-1}
            With Fisher:    Z = Y @ F @ U_s @ (U_s^T F U_s)^{-1}
            V = lstsq(X, Z)

        Step C: Fix U, V, solve D (r×r linear system)
            Without Fisher: h = diag(A^T Y B), G = (A^T A) ⊙ (B^T B)
            With Fisher:    h = diag(A^T Y F B), G = (A^T A) ⊙ (B^T F B)

        Args:
            calib_loader: Calibration data loader
            num_iters: Number of ALS iterations (default: 2)
            update_sigma: Whether to update diagonal scaling in Step C (default: True)
            token_sample_ratio: Ratio of tokens to sample per sequence to avoid OOM (default: 0.1)
            use_fisher_weight: Use Fisher information to weight output dimensions (default: False)
                              Prioritizes reconstruction accuracy for loss-sensitive outputs.
        """
        fisher_str = ", Fisher-weighted" if use_fisher_weight else ""
        print(f"Phase 4: ALS Calibration ({num_iters} iterations, update_sigma={update_sigma}, token_sample={token_sample_ratio:.0%}{fisher_str})...")

        # Move embedding layers to device
        if "opt" in self.model_name:
            self.model.model.decoder.embed_tokens = self.model.model.decoder.embed_tokens.to(self.device)
            self.model.model.decoder.embed_positions = self.model.model.decoder.embed_positions.to(self.device)
        else:
            self.model.model.embed_tokens = self.model.model.embed_tokens.to(self.device)

        # Capture inputs to first layer
        dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros(
            (len(calib_loader), self.model.seqlen, self.model.config.hidden_size),
            dtype=dtype, device='cpu'  # Store on CPU to save GPU memory
        )
        cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                # FIXED: Store inp[0] to remove batch dimension
                inps[cache['i']] = inp[0].detach().cpu().to(inps.dtype)
                cache['i'] += 1
                if cache['attention_mask'] is None:
                    cache['attention_mask'] = kwargs['attention_mask'].cpu()
                    if 'position_ids' in kwargs:
                        cache['position_ids'] = kwargs['position_ids'].cpu()
                else:
                    cache['attention_mask'] = torch.cat(
                        (cache['attention_mask'], kwargs['attention_mask'].cpu()), dim=0
                    )
                    if 'position_ids' in kwargs:
                        cache['position_ids'] = torch.cat(
                            (cache['position_ids'], kwargs['position_ids'].cpu()), dim=0
                        )
                raise ValueError

        self.layers[0] = self.layers[0].to(self.device)
        original_layer0 = self.layers[0]
        self.layers[0] = Catcher(self.layers[0])

        for batch in calib_loader:
            try:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**batch)
            except ValueError:
                pass

        self.layers[0] = original_layer0
        self.layers[0] = self.layers[0].cpu()

        # Move embedding layers back to CPU
        if "opt" in self.model_name:
            self.model.model.decoder.embed_tokens = self.model.model.decoder.embed_tokens.cpu()
            self.model.model.decoder.embed_positions = self.model.model.decoder.embed_positions.cpu()
        else:
            self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()

        torch.cuda.empty_cache()

        attention_masks = cache['attention_mask']
        position_ids = cache.get('position_ids', None)

        # Process each layer
        outs = torch.zeros_like(inps)
        total_improvement = 0.0
        calibrated_layers = 0

        # Compute number of tokens to sample per sequence
        tokens_per_seq = max(1, int(self.model.seqlen * token_sample_ratio))
        print(f"  Sampling {tokens_per_seq} tokens per sequence (total ~{len(calib_loader) * tokens_per_seq} tokens)")

        for layer_idx in tqdm(range(len(self.layers))):
            layer = self.layers[layer_idx].float().to(self.device)

            if layer_idx not in self.svd_components:
                # Just forward through this layer
                with torch.no_grad():
                    for j in range(inps.shape[0]):
                        inp_j = inps[j].unsqueeze(0).float().to(self.device)
                        mask_j = attention_masks[j].unsqueeze(0).to(self.device)
                        if position_ids is not None and "opt" not in self.model_name:
                            pos_j = position_ids[j].unsqueeze(0).to(self.device)
                            outs[j] = layer(inp_j, attention_mask=mask_j, position_ids=pos_j, use_cache=False)[0].cpu().to(dtype)
                        else:
                            outs[j] = layer(inp_j, attention_mask=mask_j, use_cache=False)[0].cpu().to(dtype)
                self.layers[layer_idx] = layer.to(dtype).cpu()
                inps = outs.clone()
                torch.cuda.empty_cache()
                continue

            # Capture ALL linear layer inputs in ONE forward pass
            # NOTE: We'll do two passes - first for attention + gate/up, then for down_proj
            # because down_proj's input depends on gate/up outputs
            subset = find_layers(layer)
            

            # ========== 1-way 分组 ==========
            # Separate projections: attention/gate/up vs down_proj
            attn_mlp_first = []  # q, k, v, o, gate, up
            mlp_down = []  # down_proj
            for name in subset:
                if name in self.svd_components[layer_idx]:
                    if 'down' not in name.lower():
                        attn_mlp_first.append(name)
                    else:
                        attn_mlp_first.append(name)


            # ========== 2-way 分组 ==========
            # Separate projections: attention/gate/up vs down_proj
            # attn_mlp_first = []  # q, k, v, o, gate, up
            # mlp_down = []  # down_proj
            # for name in subset:
            #     if name in self.svd_components[layer_idx]:
            #         if 'down' in name.lower():
            #             mlp_down.append(name)
            #         else:
            #             attn_mlp_first.append(name)


            # # ========== 4-way 分组 ==========
            # proj_all = [n for n in subset if n in self.svd_components[layer_idx]]
            
            # # 精确分组
            # # Separate projections: attention/gate/up vs down_proj
            # ATTN_QKV_LEAF = {"q_proj", "k_proj", "v_proj", "qkv_proj", "Wqkv", "query_key_value", "c_attn"}
            # ATTN_O_LEAF = {"o_proj", "out_proj"}
            # MLP_GATE_UP_LEAF = {"gate_proj", "up_proj", "fc1", "c_fc", "dense_h_to_4h"}
            # MLP_DOWN_LEAF = {"down_proj", "fc2", "dense_4h_to_h"}
            
            # ATTN_PARENT = {"self_attn", "attn", "attention", "mha"}
            # MLP_PARENT = {"mlp", "ffn", "feed_forward", "feedforward"}
            
            # attn_qkv, attn_o, mlp_gate_up, mlp_down = [], [], [], []
            
            # for name in proj_all:
            #     parts = name.split(".")
            #     leaf = parts[-1]
            #     is_attn = any(p in ATTN_PARENT for p in parts)
            #     is_mlp = any(p in MLP_PARENT for p in parts)
                
            #     if leaf in ATTN_QKV_LEAF:
            #         attn_qkv.append(name)
            #     elif leaf in ATTN_O_LEAF:
            #         attn_o.append(name)
            #     elif leaf in MLP_GATE_UP_LEAF:
            #         mlp_gate_up.append(name)
            #     elif leaf in MLP_DOWN_LEAF:
            #         mlp_down.append(name)
            #     elif is_attn:
            #         attn_o.append(name)  # fallback
            #     elif is_mlp:
            #         mlp_gate_up.append(name)  # fallback
            #     else:
            #         attn_qkv.append(name)  # fallback

            
            # Helper function to run calibration on a set of projections
            def calibrate_projections(proj_names: list, capture_fresh: bool = False):
                nonlocal layer, inps, attention_masks, position_ids

                if not proj_names:
                    return 0.0, 0  # total_improvement, count

                layer_inputs = {name: [] for name in proj_names}
                handles = []

                def make_hook(name):
                    def hook(module, inp, out):
                        x = inp[0].detach().float()
                        
                        # 处理 2D 输入
                        if x.dim() == 2:
                            T = x.shape[0]
                            if T > tokens_per_seq:
                                idx = torch.randperm(T, device=x.device)[:tokens_per_seq]
                                x = x.index_select(0, idx)
                            layer_inputs[name].append(x.cpu())
                            return
                        
                        # 处理 3D 输入：简单随机采样（与版本1一致）
                        if x.shape[1] > tokens_per_seq:
                            indices = torch.randperm(x.shape[1], device=x.device)[:tokens_per_seq]
                            x = x[:, indices, :]
                        layer_inputs[name].append(x.cpu())
                    return hook
                

                for name in proj_names:
                    linear = self._get_module_by_name(layer, name)
                    handle = linear.register_forward_hook(make_hook(name))
                    handles.append(handle)

                # Forward pass to capture inputs
                with torch.no_grad():
                    for j in range(inps.shape[0]):
                        inp_j = inps[j].unsqueeze(0).float().to(self.device)
                        mask_j = attention_masks[j].unsqueeze(0).to(self.device)
                        if position_ids is not None and "opt" not in self.model_name:
                            pos_j = position_ids[j].unsqueeze(0).to(self.device)
                            _ = layer(inp_j, attention_mask=mask_j, position_ids=pos_j, use_cache=False)
                        else:
                            _ = layer(inp_j, attention_mask=mask_j, use_cache=False)

                for handle in handles:
                    handle.remove()

                # Calibrate each projection
                proj_improvement = 0.0
                proj_count = 0

                for name in proj_names:
                    if len(layer_inputs.get(name, [])) == 0:
                        continue

                    U_r, S_r, VT_r, bias = self.svd_components[layer_idx][name]
                    rank = len(S_r)
                    original_linear = self._get_module_by_name(layer, name)

                    # Stack inputs
                    X = torch.cat([x.reshape(-1, x.shape[-1]) for x in layer_inputs[name]], dim=0).to(self.device)
                    # X = torch.cat([x.reshape(-1, x.shape[-1]) for x in layer_inputs[name]], dim=0).to(self.device).contiguous() # !!!
                    del layer_inputs[name]

                    # Teacher signal
                    W = original_linear.weight.data.float().to(self.device)
                    Y = X @ W.T
                    # Y = (X @ W.T).contiguous() # !!!

                    # SVD components
                    U = U_r.float().to(self.device)
                    S = S_r.float().to(self.device)
                    V = VT_r.T.float().to(self.device)

                    # Loss before
                    W_before = (U * S) @ VT_r.float().to(self.device)
                    loss_before = ((X @ W_before.T - Y) ** 2).mean().item()
                    del W_before

                    # Skip ALS if loss_before is already very small - nothing meaningful to optimize
                    skip_als_threshold = 1e-6
                    if loss_before < skip_als_threshold:
                        if layer_idx < 3:
                            print(f"    L{layer_idx} {name}: skipped ALS (loss_before={loss_before:.2e} < {skip_als_threshold:.0e})")
                        # Just keep original SVD, no changes needed
                        del X, W, Y, U, S, V
                        torch.cuda.empty_cache()
                        continue

                    reg = 1e-6  # Increased regularization for numerical stability
                    max_val = 1e6  # Clamp threshold to prevent value explosion

                    # Compute output-space Fisher weights (Scheme B approximation)
                    # F_y ≈ (U² @ F_σ) where F_σ is the singular value Fisher from Phase 2
                    F_y = None
                    if use_fisher_weight and hasattr(self, 'fisher_info') and layer_idx in self.fisher_info:
                        if name in self.fisher_info[layer_idx]:
                            fisher_raw = self.fisher_info[layer_idx][name]

                            # Handle different Fisher shapes
                            if fisher_raw.dim() == 1:
                                # 1D: F_σ for singular values
                                F_sigma = fisher_raw.float().to(self.device)
                            elif fisher_raw.dim() == 2:
                                # 2D: Full Fisher matrix, use diagonal or reduce
                                if fisher_raw.shape[0] == fisher_raw.shape[1]:
                                    # Square matrix, take diagonal
                                    F_sigma = fisher_raw.diag().float().to(self.device)
                                else:
                                    # Non-square, sum over one dimension
                                    F_sigma = fisher_raw.sum(dim=1).float().to(self.device)
                            else:
                                F_sigma = None

                            if F_sigma is not None and len(F_sigma) >= rank:
                                # Truncate F_sigma to match current rank (after Phase 3 truncation)
                                F_sigma = F_sigma[:rank]
                                # F_y = diag(U @ diag(F_σ) @ U^T) = (U² @ F_σ)
                                F_y = (U ** 2) @ F_sigma  # (out_dim,)
                                # Normalize to avoid numerical issues
                                F_y = F_y / (F_y.mean() + 1e-10)
                                # Clamp extreme values
                                F_y = F_y.clamp(min=0.01, max=100.0)

                    # ALS iterations
                    for als_iter in range(num_iters):
                        # Step A: Fix V, S, solve U
                        # Fisher does NOT affect Step A (cancels in the derivation)
                        Z = (X @ V) * S  # (N, r)
                        U_T_new = torch.linalg.lstsq(Z, Y).solution  # (r, out_dim)
                        U = U_T_new.T  # (out_dim, r)
                        del Z, U_T_new

                        # Step B: Fix U, S, solve V
                        # Without Fisher: Z_target = Y @ U_s @ (U_s^T @ U_s)^{-1}
                        # With Fisher:    Z_target = Y @ F @ U_s @ (U_s^T @ F @ U_s)^{-1}
                        U_s = U * S  # (out_dim, r)
                        if F_y is not None:
                            # F_y is (out_dim,), apply as diagonal: F @ U_s = U_s * F_y[:, None]
                            F_U_s = U_s * F_y.unsqueeze(1)  # (out_dim, r)
                            G = U_s.T @ F_U_s + reg * torch.eye(rank, device=self.device)  # (r, r)
                            Z_target = (Y @ F_U_s) @ torch.linalg.inv(G)  # (N, r)
                            del F_U_s
                        else:
                            G = U_s.T @ U_s + reg * torch.eye(rank, device=self.device)
                            Z_target = (Y @ U_s) @ torch.linalg.inv(G)  # (N, r)
                        V = torch.linalg.lstsq(X, Z_target).solution  # (in_dim, r)
                        del U_s, G, Z_target

                    # Step C: Fix U, V, solve D
                    # Without Fisher: h = diag(A^T @ Y @ B), G = (A^T A) ⊙ (B^T B)
                    # With Fisher:    h = diag(A^T @ Y @ F @ B), G = (A^T A) ⊙ (B^T @ F @ B)
                    if update_sigma:
                        A = X @ V  # (N, r)
                        if F_y is not None:
                            # Y @ F = Y * F_y (broadcast along columns)
                            Y_F = Y * F_y.unsqueeze(0)  # (N, out_dim)
                            YB = Y_F @ U  # (N, r)
                            # B^T @ F @ B = U^T @ diag(F_y) @ U
                            F_U = U * F_y.unsqueeze(1)  # (out_dim, r)
                            BtFB = U.T @ F_U  # (r, r)
                            del Y_F, F_U
                        else:
                            YB = Y @ U  # (N, r)
                            BtFB = U.T @ U  # (r, r)

                        h = (A * YB).sum(dim=0)  # (r,)
                        AtA = A.T @ A  # (r, r)
                        G = AtA * BtFB  # Hadamard product (r, r)

                        d = torch.linalg.solve(G + reg * torch.eye(rank, device=self.device), h)
                        S = torch.abs(d)  # Keep positive
                        del A, YB, h, AtA, BtFB, G, d

                    # Loss after
                    VT = V.T
                    W_after = (U * S) @ VT
                    loss_after = ((X @ W_after.T - Y) ** 2).mean().item()

                    # Check for NaN/Inf OR if ALS made things worse OR extreme weights - fallback to original SVD
                    use_original = False
                    if torch.isnan(W_after).any() or torch.isinf(W_after).any() or math.isnan(loss_after) or math.isinf(loss_after):
                        if layer_idx < 3:
                            print(f"    L{layer_idx} {name}: numerical issue, using original SVD")
                        use_original = True
                    elif loss_after > loss_before and loss_before > 1e-10:
                        # ALS made things worse - revert to original
                        if layer_idx < 3:
                            print(f"    L{layer_idx} {name}: ALS worsened (before={loss_before:.6f} after={loss_after:.6f}), using original SVD")
                        use_original = True
                    else:
                        # Additional check: ensure weights are not too extreme compared to original
                        W_orig = (U_r.float().to(self.device) * S_r.float().to(self.device)) @ VT_r.float().to(self.device)
                        orig_max = W_orig.abs().max().item()
                        new_max = W_after.abs().max().item()
                        # If new weights are more than 10x larger than original, revert
                        if orig_max > 0 and new_max > 10 * orig_max:
                            if layer_idx < 3:
                                print(f"    L{layer_idx} {name}: extreme weights (orig_max={orig_max:.2f}, new_max={new_max:.2f}), using original SVD")
                            use_original = True
                        del W_orig

                    if use_original:
                        # Restore original components
                        U = U_r.float().to(self.device)
                        S = S_r.float().to(self.device)
                        VT = VT_r.float().to(self.device)
                        W_after = (U * S) @ VT
                        loss_after = loss_before  # No change

                    # Protection for small loss_before
                    min_loss_threshold = 1e-10
                    if loss_before > min_loss_threshold:
                        improvement = (1 - loss_after / loss_before) * 100
                        improvement = max(-100.0, min(100.0, improvement))
                        proj_improvement += improvement
                        proj_count += 1
                        if layer_idx < 3:
                            print(f"    L{layer_idx} {name}: before={loss_before:.6f} after={loss_after:.6f} improvement={improvement:.1f}%")
                    else:
                        if layer_idx < 3:
                            print(f"    L{layer_idx} {name}: skipped (loss_before={loss_before:.2e} < threshold)")

                    # Update SVD components
                    self.svd_components[layer_idx][name] = (U.cpu(), S.cpu(), VT.cpu(),
                                                            bias.cpu() if bias is not None else None)

                    # Write back with torch.no_grad() to avoid leaf variable error
                    with torch.no_grad():
                        original_linear.weight.copy_(W_after.to(original_linear.weight.dtype))

                    del U, S, V, VT, W_after, X, W, Y
                    torch.cuda.empty_cache()

                return proj_improvement, proj_count

            # 1/2-way 分组使用的代码
            # First pass: calibrate attention and gate/up projections
            imp1, cnt1 = calibrate_projections(attn_mlp_first)
            total_improvement += imp1
            calibrated_layers += cnt1
            # Second pass: calibrate down_proj with fresh capture (after gate/up updated)
            if mlp_down:
                imp2, cnt2 = calibrate_projections(mlp_down, capture_fresh=True)
                total_improvement += imp2
                calibrated_layers += cnt2

            # 4-way 分组使用的代码
            # # 按依赖顺序校准：qkv -> o -> gate/up -> down
            # imp1, cnt1 = calibrate_projections(attn_qkv)
            # imp2, cnt2 = calibrate_projections(attn_o)
            # imp3, cnt3 = calibrate_projections(mlp_gate_up)
            # imp4, cnt4 = calibrate_projections(mlp_down)
            
            # total_improvement += imp1 + imp2 + imp3 + imp4
            # calibrated_layers += cnt1 + cnt2 + cnt3 + cnt4

            # Forward through layer for next layer's input (now using calibrated weights)
            with torch.no_grad():
                for j in range(inps.shape[0]):
                    inp_j = inps[j].unsqueeze(0).float().to(self.device)
                    mask_j = attention_masks[j].unsqueeze(0).to(self.device)
                    if position_ids is not None and "opt" not in self.model_name:
                        pos_j = position_ids[j].unsqueeze(0).to(self.device)
                        outs[j] = layer(inp_j, attention_mask=mask_j, position_ids=pos_j, use_cache=False)[0].cpu().to(dtype)
                    else:
                        outs[j] = layer(inp_j, attention_mask=mask_j, use_cache=False)[0].cpu().to(dtype)

            # Check for NaN/Inf in layer output - this indicates calibration corrupted the layer
            if torch.isnan(outs).any() or torch.isinf(outs).any():
                print(f"  ERROR: Layer {layer_idx} output contains NaN/Inf! Calibration may have corrupted weights.")
                # Replace NaN/Inf with zeros to prevent propagation (though model is likely damaged)
                outs = torch.nan_to_num(outs, nan=0.0, posinf=0.0, neginf=0.0)

            self.layers[layer_idx] = layer.to(dtype).cpu()
            inps = outs.clone()
            torch.cuda.empty_cache()

        if calibrated_layers > 0:
            avg_improvement = total_improvement / calibrated_layers
            print(f"  Average ALS improvement: {avg_improvement:.1f}% across {calibrated_layers} linear layers")
        else:
            print("  No layers calibrated")


    def phase4_als_calibration_with_blocks(self, calib_loader: List[Dict], num_iters: int = 1,
                                           token_sample_ratio: float = 0.2) -> None:
        """
        ALS calibration that accounts for residual blocks.

        When blocks are present, the total approximation is: W_approx = W_svd + W_blocks
        We want to minimize: ||X @ W_approx^T - X @ W_orig^T||²

        This is equivalent to minimizing: ||X @ W_svd^T - (X @ W_orig^T - X @ W_blocks^T)||²
        So the target for SVD becomes: Y_target = Y_orig - Y_blocks

        This allows SVD and blocks to jointly minimize reconstruction error.
        """
        if not hasattr(self, 'residual_blocks') or len(self.residual_blocks) == 0:
            print("  No blocks to account for, running standard ALS...")
            self.phase4_als_calibration(calib_loader, num_iters=num_iters,
                                        update_sigma=True, token_sample_ratio=token_sample_ratio)
            return

        print(f"Phase 4b: Joint ALS with blocks ({num_iters} iters)...")

        # Move embedding layers to device
        if "opt" in self.model_name:
            self.model.model.decoder.embed_tokens = self.model.model.decoder.embed_tokens.to(self.device)
            self.model.model.decoder.embed_positions = self.model.model.decoder.embed_positions.to(self.device)
        else:
            self.model.model.embed_tokens = self.model.model.embed_tokens.to(self.device)

        dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros(
            (len(calib_loader), self.model.seqlen, self.model.config.hidden_size),
            dtype=dtype, device='cpu'
        )
        cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp[0].detach().cpu().to(inps.dtype)
                cache['i'] += 1
                if cache['attention_mask'] is None:
                    cache['attention_mask'] = kwargs['attention_mask'].cpu()
                    cache['position_ids'] = kwargs.get('position_ids', torch.zeros(1)).cpu()
                else:
                    cache['attention_mask'] = torch.cat(
                        (cache['attention_mask'], kwargs['attention_mask'].cpu()), dim=0
                    )
                    if 'position_ids' in kwargs:
                        cache['position_ids'] = torch.cat(
                            (cache['position_ids'], kwargs['position_ids'].cpu()), dim=0
                        )
                raise ValueError

        self.layers[0] = self.layers[0].to(self.device)
        original_layer0 = self.layers[0]
        self.layers[0] = Catcher(self.layers[0])

        for batch in calib_loader:
            try:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**batch)
            except ValueError:
                pass

        self.layers[0] = original_layer0
        self.layers[0] = self.layers[0].cpu()

        if "opt" in self.model_name:
            self.model.model.decoder.embed_tokens = self.model.model.decoder.embed_tokens.cpu()
            self.model.model.decoder.embed_positions = self.model.model.decoder.embed_positions.cpu()
        else:
            self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()

        torch.cuda.empty_cache()

        attention_masks = cache['attention_mask']
        position_ids = cache.get('position_ids', None)
        tokens_per_seq = max(1, int(self.model.seqlen * token_sample_ratio))

        total_improvement = 0.0
        calibrated_count = 0
        outs = torch.zeros_like(inps)

        for layer_idx in tqdm(range(len(self.layers)), desc="Joint ALS"):
            layer = self.layers[layer_idx].float().to(self.device)

            if layer_idx not in self.svd_components:
                with torch.no_grad():
                    for j in range(inps.shape[0]):
                        inp_j = inps[j].unsqueeze(0).float().to(self.device)
                        mask_j = attention_masks[j].unsqueeze(0).to(self.device)
                        if position_ids is not None and "opt" not in self.model_name:
                            pos_j = position_ids[j].unsqueeze(0).to(self.device)
                            outs[j] = layer(inp_j, attention_mask=mask_j, position_ids=pos_j, use_cache=False)[0].cpu().to(dtype)
                        else:
                            outs[j] = layer(inp_j, attention_mask=mask_j, use_cache=False)[0].cpu().to(dtype)
                self.layers[layer_idx] = layer.to(dtype).cpu()
                inps = outs.clone()
                torch.cuda.empty_cache()
                continue

            subset = find_layers(layer)
            layer_inputs = {name: [] for name in subset}
            handles = []

            def make_hook(name):
                def hook(module, inp, out):
                    x = inp[0].detach().float()
                    if x.dim() == 2:
                        T = x.shape[0]
                        if T > tokens_per_seq:
                            idx = torch.randperm(T, device=x.device)[:tokens_per_seq]
                            x = x.index_select(0, idx)
                        layer_inputs[name].append(x.cpu())
                        return
                    if x.shape[1] > tokens_per_seq:
                        indices = torch.randperm(x.shape[1], device=x.device)[:tokens_per_seq]
                        x = x[:, indices, :]
                    layer_inputs[name].append(x.cpu())
                return hook

            for name in subset:
                handle = subset[name].register_forward_hook(make_hook(name))
                handles.append(handle)

            with torch.no_grad():
                for j in range(inps.shape[0]):
                    inp_j = inps[j].unsqueeze(0).float().to(self.device)
                    mask_j = attention_masks[j].unsqueeze(0).to(self.device)
                    if position_ids is not None and "opt" not in self.model_name:
                        pos_j = position_ids[j].unsqueeze(0).to(self.device)
                        _ = layer(inp_j, attention_mask=mask_j, position_ids=pos_j, use_cache=False)
                    else:
                        _ = layer(inp_j, attention_mask=mask_j, use_cache=False)

            for handle in handles:
                handle.remove()

            # Calibrate each projection with block adjustment
            for name in subset:
                key = (layer_idx, name)
                if name not in self.svd_components[layer_idx] or len(layer_inputs.get(name, [])) == 0:
                    continue

                U_r, S_r, VT_r, bias = self.svd_components[layer_idx][name]
                rank = len(S_r)
                original_linear = self._get_module_by_name(layer, name)

                X = torch.cat([x.reshape(-1, x.shape[-1]) for x in layer_inputs[name]], dim=0).to(self.device)
                del layer_inputs[name]

                # Original target
                W_orig = original_linear.weight.data.float().to(self.device)
                Y_orig = X @ W_orig.T

                # Compute block contribution and subtract from target
                Y_blocks = torch.zeros_like(Y_orig)
                if key in self.residual_blocks:
                    for block in self.residual_blocks[key]:
                        row, col = block['row'], block['col']
                        row_end, col_end = block['row_end'], block['col_end']
                        B_val = block['val'].float().to(self.device)
                        # Block contribution: X[:, col:col_end] @ B_val.T -> Y[:, row:row_end]
                        Y_blocks[:, row:row_end] += X[:, col:col_end] @ B_val.T

                # Adjusted target: what SVD needs to approximate
                Y = Y_orig - Y_blocks

                # SVD components
                U = U_r.float().to(self.device)
                S = S_r.float().to(self.device)
                V = VT_r.T.float().to(self.device)

                # Loss before
                W_before = (U * S) @ VT_r.float().to(self.device)
                loss_before = ((X @ W_before.T - Y) ** 2).mean().item()

                if loss_before < 1e-6:
                    del X, W_orig, Y_orig, Y_blocks, Y, U, S, V, W_before
                    torch.cuda.empty_cache()
                    continue

                reg = 1e-6

                # ALS iterations
                for _ in range(num_iters):
                    # Step A: Fix V, S, solve U
                    Z = (X @ V) * S
                    U_T_new = torch.linalg.lstsq(Z, Y).solution
                    U = U_T_new.T
                    del Z, U_T_new

                    # Step B: Fix U, S, solve V
                    U_s = U * S
                    G = U_s.T @ U_s + reg * torch.eye(rank, device=self.device)
                    Z_target = (Y @ U_s) @ torch.linalg.inv(G)
                    V = torch.linalg.lstsq(X, Z_target).solution
                    del U_s, G, Z_target

                # Step C: Solve D
                A = X @ V
                YB = Y @ U
                h = (A * YB).sum(dim=0)
                AtA = A.T @ A
                BtB = U.T @ U
                G = AtA * BtB
                d = torch.linalg.solve(G + reg * torch.eye(rank, device=self.device), h)
                S = torch.abs(d)
                del A, YB, h, AtA, BtB, G, d

                # Loss after
                VT = V.T
                W_after = (U * S) @ VT
                loss_after = ((X @ W_after.T - Y) ** 2).mean().item()

                # Check for issues
                if torch.isnan(W_after).any() or loss_after > loss_before:
                    U = U_r.float().to(self.device)
                    S = S_r.float().to(self.device)
                    VT = VT_r.float().to(self.device)
                else:
                    improvement = (1 - loss_after / loss_before) * 100
                    total_improvement += improvement
                    calibrated_count += 1

                    # Update components
                    self.svd_components[layer_idx][name] = (U.cpu(), S.cpu(), VT.cpu(),
                                                            bias.cpu() if bias is not None else None)
                    # Note: linear weight will be updated during apply_compression
                    # For now, update with SVD approximation for subsequent layer forward passes
                    with torch.no_grad():
                        original_linear.weight.copy_(W_after.to(original_linear.weight.dtype))

                del U, S, V, VT, W_after, X, W_orig, Y_orig, Y_blocks, Y
                torch.cuda.empty_cache()

            # Forward through layer
            with torch.no_grad():
                for j in range(inps.shape[0]):
                    inp_j = inps[j].unsqueeze(0).float().to(self.device)
                    mask_j = attention_masks[j].unsqueeze(0).to(self.device)
                    if position_ids is not None and "opt" not in self.model_name:
                        pos_j = position_ids[j].unsqueeze(0).to(self.device)
                        outs[j] = layer(inp_j, attention_mask=mask_j, position_ids=pos_j, use_cache=False)[0].cpu().to(dtype)
                    else:
                        outs[j] = layer(inp_j, attention_mask=mask_j, use_cache=False)[0].cpu().to(dtype)

            self.layers[layer_idx] = layer.to(dtype).cpu()
            inps = outs.clone()
            torch.cuda.empty_cache()

        if calibrated_count > 0:
            print(f"  Joint ALS improvement: {total_improvement/calibrated_count:.1f}% avg across {calibrated_count} layers")


def fisher_aware_svd_compression(model_name: str, model: nn.Module,
                                  calib_loader: List[Dict], ratio: float,
                                  whitening_mat: Optional[Dict] = None,
                                  device: str = "cuda",
                                  use_low_resource: bool = False,
                                  num_gpus: int = 1,
                                  calibration_steps: int = 50,
                                  min_rank: int = 16,
                                  fisher_lambda: float = 2.0,
                                  use_als: bool = True,
                                  als_iters: int = 2,
                                  token_sample_ratio: float = 0.2) -> nn.Module:
    """
    Main entry point for Fisher-Aware SVD compression.

    Uses adaptive min/max rank allocation:
    - f_min: Binary search to achieve target floor_share of budget
    - max_factor: Per-projection, based on score entropy/concentration
      (sharp distributions get higher max, flat distributions get lower max)

    Args:
        model_name: Name of the model (e.g., "llama", "mistral", "opt")
        model: The model to compress
        calib_loader: Calibration data loader
        ratio: Target compression ratio (0-1). Higher means more parameters kept.
        whitening_mat: Optional whitening matrices from SVD-LLM profiling
        device: Device to use for computation
        use_low_resource: Use memory-efficient proxy loss (default: False, use true CE loss)
        num_gpus: Number of GPUs to use for model parallelism (default: 1)
        calibration_steps: Number of Phase 4 calibration steps (default: 50)
        min_rank: Minimum rank to keep per projection (default: 16)
        fisher_lambda: Weight for Fisher in log-space formula (default: 2.0)
                      Formula: Score = log(σ) + λ × log(F)
                      Higher values give Fisher more influence on ranking.
        use_als: Use ALS calibration instead of M-optimization (default: True)
        als_iters: Number of ALS iterations per layer (default: 2)
        token_sample_ratio: Ratio of tokens to sample per sequence for ALS (default: 0.1)

    Returns:
        Compressed model
    """
    print(f"Fisher-Aware SVD Compression")
    print(f"  Mode: {'Proxy Loss (low resource)' if use_low_resource else 'Cross-Entropy Loss (full)'}")
    print(f"  GPUs: {num_gpus}")
    print(f"  Fisher λ: {fisher_lambda} (log-space formula)")
    print(f"  Min rank: {min_rank} (adaptive f_min and max_factor)")
    print(f"  Phase 4: {'ALS' if use_als else 'M-optimization'} ({als_iters} iterations, {token_sample_ratio:.0%} tokens)" if use_als else "  Phase 4: M-optimization")

    compressor = FisherAwareSVD(model, model_name, device, num_gpus=num_gpus)
    return compressor.compress(calib_loader, ratio, whitening_mat, use_low_resource,
                               calibration_steps, min_rank=min_rank, fisher_lambda=fisher_lambda,
                               use_als=use_als, als_iters=als_iters,
                               token_sample_ratio=token_sample_ratio)


if __name__ == '__main__':
    import argparse
    from evaluater import ppl_eval

    parser = argparse.ArgumentParser(description="Fisher-Aware SVD Compression")
    parser.add_argument('--model', type=str, required=True, help='Model name or path')
    parser.add_argument('--ratio', type=float, default=0.2,
                       help='Compression ratio (0-1), default=0.2 means keep 20%% params')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                       help='Calibration dataset [wikitext2, ptb, c4]')
    parser.add_argument('--nsamples', type=int, default=256,
                       help='Number of calibration samples')
    parser.add_argument('--seqlen', type=int, default=2048,
                       help='Sequence length')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save compressed model')
    parser.add_argument('--use_whitening', action='store_true',
                       help='Use whitening matrices (requires profiling)')
    parser.add_argument('--eval', action='store_true',
                       help='Evaluate perplexity after compression')

    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer = get_model_from_huggingface(args.model)
    model = model.eval()

    # Load calibration data
    print(f"Loading calibration data from {args.dataset}...")
    calib_loader = get_calib_train_data(
        args.dataset, tokenizer, args.nsamples, seqlen=args.seqlen
    )

    # Optionally get whitening matrices
    whitening_mat = None
    if args.use_whitening:
        from SVDLLM import profle_svdllm_low_resource
        print("Computing whitening matrices...")
        whitening_mat = profle_svdllm_low_resource(args.model, model, calib_loader, args.device)

    # Compress
    ratio = 1 - args.ratio  # Convert to retention ratio
    print(f"\nStarting Fisher-Aware SVD compression (retention ratio: {ratio:.2%})...")

    model = fisher_aware_svd_compression(
        args.model, model, calib_loader, ratio,
        whitening_mat=whitening_mat,
        device=args.device,
        use_low_resource=True
    )

    # Save if requested
    if args.save_path:
        print(f"Saving compressed model to {args.save_path}")
        os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
        torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path)

    # Evaluate if requested
    if args.eval:
        print("\nEvaluating perplexity...")
        model = model.float().to(args.device)
        ppl_eval(model, tokenizer, datasets=['wikitext2'],
                model_seq_len=args.seqlen, batch_size=4, device=args.device)
