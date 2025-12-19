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
        out = torch.matmul(x, self.VT.T)  # x @ V
        out = out * self.sigma  # element-wise multiply with sigma
        out = torch.matmul(out, self.U.T)  # @ U^T
        if self.bias is not None:
            out = out + self.bias
        return out


class FisherAwareSVD:
    """
    Fisher-Aware SVD compression for LLMs.

    This class implements the three-phase algorithm:
    1. Phase 1: SVD decomposition of each linear layer
    2. Phase 2: Sensitivity estimation via empirical Fisher information
    3. Phase 3: Global truncation based on importance scores
    """

    def __init__(self, model: nn.Module, model_name: str, device: str = "cuda"):
        self.model = model
        self.model_name = model_name
        self.device = device

        # Get layers based on model type
        if "opt" in model_name:
            self.layers = model.model.decoder.layers
        else:
            self.layers = model.model.layers

        # Storage for SVD components and Fisher information
        self.svd_components: Dict[str, Dict[str, Tuple[torch.Tensor, ...]]] = {}
        self.fisher_info: Dict[str, Dict[str, torch.Tensor]] = {}
        self.original_layers: Dict[str, nn.Module] = {}

    def phase1_svd_decomposition(self, whitening_mat: Optional[Dict] = None) -> None:
        """
        Phase 1: Perform SVD decomposition on each linear layer.

        Args:
            whitening_mat: Optional whitening matrices from SVD-LLM profiling.
                          If provided, applies whitening before SVD.
        """
        print("Phase 1: SVD Decomposition...")

        for layer_idx in tqdm(range(len(self.layers))):
            layer = self.layers[layer_idx]
            subset = find_layers(layer)

            layer_svd = {}
            for name, module in subset.items():
                W = module.weight.data.float()

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
        for layer_idx in range(len(self.layers)):
            layer = self.layers[layer_idx]
            subset = find_layers(layer)

            for name, module in subset.items():
                if layer_idx in self.svd_components and name in self.svd_components[layer_idx]:
                    U, S, VT, bias = self.svd_components[layer_idx][name]
                    svd_layer = SVDParameterizedLinear(
                        U.to(self.device),
                        S.to(self.device),
                        VT.to(self.device),
                        bias.to(self.device) if bias is not None else None
                    )

                    # Replace the layer
                    self._set_module_by_name(layer, name, svd_layer)

    def _restore_original_layers(self) -> None:
        """Restore original linear layers from SVD components."""
        for layer_idx in range(len(self.layers)):
            layer = self.layers[layer_idx]
            subset = find_layers(layer)

            for name in subset.keys():
                if layer_idx in self.svd_components and name in self.svd_components[layer_idx]:
                    U, S, VT, bias = self.svd_components[layer_idx][name]
                    # Reconstruct W = U @ diag(S) @ VT
                    W = torch.matmul(U * S, VT)

                    # Create new linear layer
                    out_features, in_features = W.shape
                    new_linear = nn.Linear(in_features, out_features, bias=bias is not None)
                    new_linear.weight.data = W.to(subset[name].weight.dtype)
                    if bias is not None:
                        new_linear.bias.data = bias.to(subset[name].weight.dtype)

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
        Low-resource Fisher estimation using true cross-entropy loss.

        This method computes Fisher information by:
        1. For each layer l, replace with SVD-parameterized version
        2. Forward through remaining layers to get logits
        3. Compute cross-entropy loss and backpropagate
        4. Accumulate squared gradients as Fisher information

        Memory efficient: processes one layer at a time while still using true task loss.
        """

        print("  Using layer-wise estimation with cross-entropy loss...")

        # Move all components to device for end-to-end forward pass
        if "opt" in self.model_name:
            self.model.model.decoder.embed_tokens = self.model.model.decoder.embed_tokens.to(self.device)
            self.model.model.decoder.final_layer_norm = self.model.model.decoder.final_layer_norm.to(self.device)
            self.model.model.decoder.embed_positions = self.model.model.decoder.embed_positions.to(self.device)
        else:
            self.model.model.embed_tokens = self.model.model.embed_tokens.to(self.device)
            self.model.model.norm = self.model.model.norm.to(self.device)

        # Also need lm_head for loss computation
        if hasattr(self.model, 'lm_head'):
            self.model.lm_head = self.model.lm_head.to(self.device)

        # Capture inputs to first layer and store input_ids for loss computation
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
            self.model.model.decoder.final_layer_norm = self.model.model.decoder.final_layer_norm.cpu()
            self.model.model.decoder.embed_positions = self.model.model.decoder.embed_positions.cpu()
        else:
            self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()
            self.model.model.norm = self.model.model.norm.cpu()

        if hasattr(self.model, 'lm_head'):
            self.model.lm_head = self.model.lm_head.cpu()

        torch.cuda.empty_cache()

        attention_masks = cache['attention_mask']
        position_ids = cache.get('position_ids', None)
        input_ids_tensor = torch.cat(input_ids_list, dim=0)

        # Process each layer with sensitivity estimation
        outs = torch.zeros_like(inps)
        total_loss = 0.0
        num_forward = 0

        for layer_idx in tqdm(range(len(self.layers))):
            layer = self.layers[layer_idx].to(self.device)
            subset = find_layers(layer)

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

            # Move remaining layers and lm_head to device for forward pass
            remaining_layers_on_device = []
            for k in range(layer_idx + 1, len(self.layers)):
                self.layers[k] = self.layers[k].to(self.device)
                remaining_layers_on_device.append(k)

            if "opt" in self.model_name:
                self.model.model.decoder.final_layer_norm = self.model.model.decoder.final_layer_norm.to(self.device)
            else:
                self.model.model.norm = self.model.model.norm.to(self.device)

            if hasattr(self.model, 'lm_head'):
                self.model.lm_head = self.model.lm_head.to(self.device)

            # Process each sample
            for j in range(inps.shape[0]):
                # Zero gradients
                for svd_layer in svd_layers.values():
                    if svd_layer.sigma.grad is not None:
                        svd_layer.sigma.grad.zero_()

                # Forward pass through current layer
                inp_j = inps[j].unsqueeze(0).clone()
                if position_ids is not None and "opt" not in self.model_name:
                    hidden = layer(inp_j,
                                   attention_mask=attention_masks[j].unsqueeze(0),
                                   position_ids=position_ids[j].unsqueeze(0))[0]
                else:
                    hidden = layer(inp_j,
                                   attention_mask=attention_masks[j].unsqueeze(0))[0]

                # Forward through remaining layers
                for k in range(layer_idx + 1, len(self.layers)):
                    if position_ids is not None and "opt" not in self.model_name:
                        hidden = self.layers[k](hidden,
                                                attention_mask=attention_masks[j].unsqueeze(0),
                                                position_ids=position_ids[j].unsqueeze(0))[0]
                    else:
                        hidden = self.layers[k](hidden,
                                                attention_mask=attention_masks[j].unsqueeze(0))[0]

                # Apply final layer norm
                if "opt" in self.model_name:
                    hidden = self.model.model.decoder.final_layer_norm(hidden)
                else:
                    hidden = self.model.model.norm(hidden)

                # Compute logits and cross-entropy loss
                if hasattr(self.model, 'lm_head'):
                    logits = self.model.lm_head(hidden)
                else:
                    logits = hidden

                # Compute cross-entropy loss (Algorithm: L ← CrossEntropy(M(x)))
                labels = input_ids_tensor[j].unsqueeze(0).to(self.device)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                total_loss += loss.item()
                num_forward += 1

                # Backward pass to get gradients w.r.t. sigma
                loss.backward()

                # Accumulate squared gradients (Fisher information)
                # F_σ^(l) ← F_σ^(l) + (g_Σ^(l))²
                for name, svd_layer in svd_layers.items():
                    if svd_layer.sigma.grad is not None:
                        layer_fisher[name] += svd_layer.sigma.grad.pow(2).cpu()

                # Store output for next layer (without gradients)
                with torch.no_grad():
                    if position_ids is not None and "opt" not in self.model_name:
                        outs[j] = layer(inps[j].unsqueeze(0),
                                        attention_mask=attention_masks[j].unsqueeze(0),
                                        position_ids=position_ids[j].unsqueeze(0))[0]
                    else:
                        outs[j] = layer(inps[j].unsqueeze(0),
                                        attention_mask=attention_masks[j].unsqueeze(0))[0]

            # Average Fisher information: F_σ^(l) ← F_σ^(l) / |D|
            for name in layer_fisher:
                layer_fisher[name] /= inps.shape[0]

            self.fisher_info[layer_idx] = layer_fisher

            # Move remaining layers back to CPU
            for k in remaining_layers_on_device:
                self.layers[k] = self.layers[k].cpu()

            if "opt" in self.model_name:
                self.model.model.decoder.final_layer_norm = self.model.model.decoder.final_layer_norm.cpu()
            else:
                self.model.model.norm = self.model.model.norm.cpu()

            if hasattr(self.model, 'lm_head'):
                self.model.lm_head = self.model.lm_head.cpu()

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

        avg_loss = total_loss / num_forward if num_forward > 0 else 0
        print(f"  Average cross-entropy loss: {avg_loss:.4f}")
        print(f"  Estimated Fisher information for {len(self.layers)} layers")

    def _estimate_fisher_full(self, calib_loader: List[Dict]) -> None:
        """
        Full Fisher estimation using end-to-end backpropagation with true task loss.

        This method computes the exact Fisher information by:
        1. Replacing all linear layers with SVD-parameterized versions
        2. Running forward pass to compute cross-entropy loss
        3. Backpropagating to get gradients w.r.t. all singular values
        4. Accumulating squared gradients as Fisher information

        This is the most accurate method but requires more GPU memory.
        """

        print("  Using end-to-end task loss (cross-entropy) for Fisher estimation...")

        # Replace all layers with SVD-parameterized versions
        self._replace_with_svd_layers()

        self.model = self.model.to(self.device)
        self.model.train()  # Enable gradient computation

        # Initialize Fisher accumulators
        for layer_idx in range(len(self.layers)):
            layer = self.layers[layer_idx]
            subset = find_layers(layer)
            layer_fisher = {}
            for name in subset:
                if isinstance(subset[name], SVDParameterizedLinear):
                    layer_fisher[name] = torch.zeros_like(subset[name].sigma)
            self.fisher_info[layer_idx] = layer_fisher

        # Accumulate Fisher information
        num_samples = 0
        total_loss = 0.0

        for batch in tqdm(calib_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Zero gradients
            self.model.zero_grad()

            # Forward pass with cross-entropy loss
            # Use labels = input_ids for language modeling
            outputs = self.model(**batch, labels=batch['input_ids'])
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Accumulate squared gradients (Fisher information)
            for layer_idx in range(len(self.layers)):
                layer = self.layers[layer_idx]
                subset = find_layers(layer)
                for name in subset:
                    if isinstance(subset[name], SVDParameterizedLinear):
                        if subset[name].sigma.grad is not None:
                            self.fisher_info[layer_idx][name] += subset[name].sigma.grad.pow(2).cpu()

            num_samples += 1

        # Average Fisher information
        for layer_idx in self.fisher_info:
            for name in self.fisher_info[layer_idx]:
                self.fisher_info[layer_idx][name] /= num_samples

        avg_loss = total_loss / num_samples
        print(f"  Average calibration loss: {avg_loss:.4f}")

        # Restore original model
        self._restore_original_layers()
        self.model = self.model.cpu()
        self.model.eval()

        print(f"  Estimated Fisher information using {num_samples} samples")

    def compute_importance_scores(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute importance scores for all singular values.

        Score_i = σ_i² × F_ii

        Returns:
            Dictionary of importance scores per layer and sublayer
        """
        importance_scores = {}

        for layer_idx in self.svd_components:
            layer_scores = {}
            for name in self.svd_components[layer_idx]:
                U, S, VT, bias = self.svd_components[layer_idx][name]

                if layer_idx in self.fisher_info and name in self.fisher_info[layer_idx]:
                    F = self.fisher_info[layer_idx][name]
                    # Score_i = σ_i² × F_ii
                    scores = S.pow(2) * F
                else:
                    # Fallback to magnitude-based scoring
                    scores = S.pow(2)

                layer_scores[name] = scores
            importance_scores[layer_idx] = layer_scores

        return importance_scores

    def phase3_global_truncation(self, ratio: float) -> None:
        """
        Phase 3: Global truncation based on importance scores.

        Following the algorithm:
        1. Compute Score_i^(l) = (σ_i^(l))² × F_σi^(l) for all singular values
        2. Flatten all scores into a list S and sort globally (descending)
        3. Keep top ρ proportion of scores, zero out the rest
        4. Reconstruct W'^(l) = U^(l) Σ'^(l) V^(l)^T

        Args:
            ratio: Target retention ratio (0-1). Higher means more parameters kept.
        """
        print(f"Phase 3: Global Truncation (target ratio: {ratio:.2%})...")

        # Compute importance scores: Score_i = σ_i² × F_ii
        importance_scores = self.compute_importance_scores()

        # Collect all scores with their identifiers (layer_idx, name, singular_value_idx)
        all_scores = []
        for layer_idx in importance_scores:
            for name in importance_scores[layer_idx]:
                scores = importance_scores[layer_idx][name]
                for i, score in enumerate(scores):
                    all_scores.append((score.item(), layer_idx, name, i))

        # Sort by importance (descending) - following Algorithm line 24
        all_scores.sort(key=lambda x: x[0], reverse=True)

        # Calculate total singular values and target count
        total_sv_count = len(all_scores)

        # Calculate per-layer parameter constraints for proper compression ratio
        # For W ∈ R^{m×n} with rank r: params = r(m+n), original = mn
        # To achieve compression ratio ρ: r(m+n) ≈ ρ × mn → r ≈ ρmn/(m+n)
        layer_max_rank = {}
        total_original_params = 0
        for layer_idx in self.svd_components:
            for name in self.svd_components[layer_idx]:
                U, S, VT, _ = self.svd_components[layer_idx][name]
                m, n = U.shape[0], VT.shape[1]
                total_original_params += m * n
                # Maximum rank that satisfies compression ratio
                max_rank = int(m * n * ratio / (m + n))
                max_rank = max(1, min(max_rank, len(S)))
                layer_max_rank[(layer_idx, name)] = max_rank

        # Total target singular values (sum of per-layer max ranks)
        total_target_sv = sum(layer_max_rank.values())

        # Keep top singular values globally (Algorithm line 25)
        # Pure global selection - no per-layer constraints beyond max rank
        kept_indices: Dict[int, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
        kept_count = 0

        for score, layer_idx, name, idx in all_scores:
            if kept_count >= total_target_sv:
                break

            current_kept = len(kept_indices[layer_idx][name])
            max_for_layer = layer_max_rank.get((layer_idx, name), 0)

            # Pure global selection: only check if we haven't exceeded max rank for this layer
            if current_kept < max_for_layer:
                kept_indices[layer_idx][name].add(idx)
                kept_count += 1

        # Truncate SVD components (Algorithm lines 26-28)
        for layer_idx in self.svd_components:
            for name in self.svd_components[layer_idx]:
                U, S, VT, bias = self.svd_components[layer_idx][name]

                # Get indices to keep, sorted by original order
                if layer_idx in kept_indices and name in kept_indices[layer_idx]:
                    indices = sorted(list(kept_indices[layer_idx][name]))
                else:
                    # Fallback: keep at least one singular value
                    indices = [0]

                if len(indices) == 0:
                    indices = [0]

                indices = torch.tensor(indices)

                # Truncate: keep only selected singular values
                U_trunc = U[:, indices]
                S_trunc = S[indices]
                VT_trunc = VT[indices, :]

                self.svd_components[layer_idx][name] = (U_trunc, S_trunc, VT_trunc, bias)

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
        print(f"  Actual compression ratio: {actual_ratio:.2%}")
        print(f"  Kept {kept_count} singular values out of {total_sv_count}")

    def apply_compression(self, ratio: float) -> None:
        """
        Apply compression to the model by replacing layers with SVD-factorized versions.

        Args:
            ratio: Compression ratio (0-1)
        """
        print("Applying compression to model...")

        for layer_idx in tqdm(range(len(self.layers))):
            layer = self.layers[layer_idx]
            subset = find_layers(layer)

            # Create SVD-based replacement modules
            if "llama" in self.model_name or "vicuna" in self.model_name:
                svd_attn = SVD_LlamaAttention(config=self.model.config, ratio=ratio)
                svd_mlp = SVD_LlamaMLP(
                    hidden_size=layer.hidden_size,
                    intermediate_size=self.model.config.intermediate_size,
                    hidden_act=self.model.config.hidden_act,
                    ratio=ratio
                )
            elif "mistral" in self.model_name:
                svd_attn = SVD_MistralAttention(config=self.model.config, ratio=ratio)
                svd_mlp = SVD_MistralMLP(config=self.model.config, ratio=ratio)
            elif 'opt' in self.model_name:
                svd_decoder = SVDOPTDecoderLayer(self.model.config, ratio=ratio)

            dtype = next(iter(self.model.parameters())).dtype

            for name in subset:
                if layer_idx not in self.svd_components or name not in self.svd_components[layer_idx]:
                    continue

                U, S, VT, bias = self.svd_components[layer_idx][name]

                # Compute U' = U @ sqrt(Sigma) and V' = sqrt(Sigma) @ V
                sqrt_sigma = torch.sqrt(torch.diag(S))
                svd_u = torch.matmul(U, sqrt_sigma).to(dtype)
                svd_v = torch.matmul(sqrt_sigma, VT).to(dtype)

                # Assign to appropriate module
                if 'opt' in self.model_name:
                    if "q_proj" in name:
                        svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                        svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                        if bias is not None:
                            svd_decoder.self_attn.q_u_proj.bias.data = bias.to(dtype)
                    elif "k_proj" in name:
                        svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                        svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                        if bias is not None:
                            svd_decoder.self_attn.k_u_proj.bias.data = bias.to(dtype)
                    elif "v_proj" in name:
                        svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                        svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                        if bias is not None:
                            svd_decoder.self_attn.v_u_proj.bias.data = bias.to(dtype)
                    elif "out_proj" in name:
                        svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                        svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                        if bias is not None:
                            svd_decoder.self_attn.out_u_proj.bias.data = bias.to(dtype)
                    elif "fc1" in name:
                        svd_decoder.fc1_u_proj.weight.data = svd_u
                        svd_decoder.fc1_v_proj.weight.data = svd_v
                        if bias is not None:
                            svd_decoder.fc1_u_proj.bias.data = bias.to(dtype)
                    elif "fc2" in name:
                        svd_decoder.fc2_u_proj.weight.data = svd_u
                        svd_decoder.fc2_v_proj.weight.data = svd_v
                        if bias is not None:
                            svd_decoder.fc2_u_proj.bias.data = bias.to(dtype)
                        svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                        svd_decoder.final_layer_norm = layer.final_layer_norm
                        self.layers[layer_idx] = svd_decoder
                else:
                    if "q_proj" in name:
                        svd_attn.q_u_proj.weight.data = svd_u
                        svd_attn.q_v_proj.weight.data = svd_v
                    elif "k_proj" in name:
                        svd_attn.k_u_proj.weight.data = svd_u
                        svd_attn.k_v_proj.weight.data = svd_v
                    elif "v_proj" in name:
                        svd_attn.v_u_proj.weight.data = svd_u
                        svd_attn.v_v_proj.weight.data = svd_v
                    elif "o_proj" in name:
                        svd_attn.o_u_proj.weight.data = svd_u
                        svd_attn.o_v_proj.weight.data = svd_v
                        layer.self_attn = svd_attn
                    elif "gate_proj" in name:
                        svd_mlp.gate_u_proj.weight.data = svd_u
                        svd_mlp.gate_v_proj.weight.data = svd_v
                    elif "down_proj" in name:
                        svd_mlp.down_u_proj.weight.data = svd_u
                        svd_mlp.down_v_proj.weight.data = svd_v
                    elif "up_proj" in name:
                        svd_mlp.up_u_proj.weight.data = svd_u
                        svd_mlp.up_v_proj.weight.data = svd_v
                        layer.mlp = svd_mlp

            torch.cuda.empty_cache()

        print("  Compression applied successfully")

    def compress(self, calib_loader: List[Dict], ratio: float,
                 whitening_mat: Optional[Dict] = None,
                 use_low_resource: bool = True) -> nn.Module:
        """
        Full compression pipeline.

        Args:
            calib_loader: Calibration data loader
            ratio: Target compression ratio (0-1)
            whitening_mat: Optional whitening matrices from SVD-LLM
            use_low_resource: Use memory-efficient processing

        Returns:
            Compressed model
        """
        # Phase 1: SVD Decomposition
        self.phase1_svd_decomposition(whitening_mat)

        # Phase 2: Sensitivity Estimation
        self.phase2_sensitivity_estimation(calib_loader, use_low_resource)

        # Phase 3: Global Truncation
        self.phase3_global_truncation(ratio)

        # Apply compression to model
        self.apply_compression(ratio)

        return self.model


def fisher_aware_svd_compression(model_name: str, model: nn.Module,
                                  calib_loader: List[Dict], ratio: float,
                                  whitening_mat: Optional[Dict] = None,
                                  device: str = "cuda",
                                  use_low_resource: bool = True) -> nn.Module:
    """
    Main entry point for Fisher-Aware SVD compression.

    Args:
        model_name: Name of the model (e.g., "llama", "mistral", "opt")
        model: The model to compress
        calib_loader: Calibration data loader
        ratio: Target compression ratio (0-1). Higher means more parameters kept.
        whitening_mat: Optional whitening matrices from SVD-LLM profiling
        device: Device to use for computation
        use_low_resource: Use memory-efficient layer-by-layer processing

    Returns:
        Compressed model
    """
    compressor = FisherAwareSVD(model, model_name, device)
    return compressor.compress(calib_loader, ratio, whitening_mat, use_low_resource)


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
