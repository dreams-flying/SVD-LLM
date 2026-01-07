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
                                   fisher_lambda: float = 2.0) -> None:
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
        """
        print(f"Phase 3: Global Truncation (target ratio: {ratio:.2%}, min_rank: {min_rank}, λ={fisher_lambda})...")

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
        target_params = int(total_original_params * ratio)
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
        print(f"  Actual compression ratio: {actual_ratio:.2%}")
        print(f"  Kept {kept_count} singular values out of {total_sv_count}")

    def apply_compression(self, ratio: float) -> None:
        """
        Apply compression to the model by replacing layers with SVD-factorized versions.

        Note: Since global truncation produces different ranks per layer/projection,
        we need to create SVD modules with the actual truncated ranks, not a uniform ratio.

        Args:
            ratio: Compression ratio (0-1) - used only for module creation reference
        """
        print("Applying compression to model...")

        # First, compute actual ranks for each layer to determine per-layer ratios
        layer_ranks = {}
        total_original_params = 0
        total_compressed_params = 0

        for layer_idx in self.svd_components:
            layer_ranks[layer_idx] = {}
            for name in self.svd_components[layer_idx]:
                U, S, VT, bias = self.svd_components[layer_idx][name]
                actual_rank = len(S)
                layer_ranks[layer_idx][name] = actual_rank
                m, n = U.shape[0], VT.shape[1]
                total_original_params += m * n
                total_compressed_params += actual_rank * (m + n)

        # Print compression summary
        print(f"  Original params: {total_original_params:,}")
        print(f"  Compressed params: {total_compressed_params:,}")
        print(f"  Compression ratio: {total_compressed_params / total_original_params:.2%}")

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

                # Replace in model using a wrapper or direct replacement
                self._replace_linear_with_svd(layer, name, u_proj, v_proj, layer_idx)
                replaced_count += 1

            torch.cuda.empty_cache()

        print(f"  Replaced {replaced_count} linear layers with SVD factorization")

    def _replace_linear_with_svd(self, layer, name: str, u_proj: nn.Linear,
                                  v_proj: nn.Linear, layer_idx: int) -> None:
        """
        Replace a linear layer with SVD factorization (V @ U).

        Args:
            layer: The transformer layer
            name: Name of the linear layer (e.g., "self_attn.q_proj")
            u_proj: The U projection (rank -> out_features)
            v_proj: The V projection (in_features -> rank)
            layer_idx: Layer index for model-specific handling
        """
        # Use module-level SVDLinear class for pickle compatibility

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
                 token_sample_ratio: float = 0.2) -> nn.Module:
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

        Returns:
            Compressed model
        """
        # Phase 1: SVD Decomposition
        self.phase1_svd_decomposition(whitening_mat)

        # Phase 2: Sensitivity Estimation
        self.phase2_sensitivity_estimation(calib_loader, use_low_resource)

        # Phase 3: Global Truncation with adaptive min/max allocation
        self.phase3_global_truncation(ratio, min_rank=min_rank, fisher_lambda=fisher_lambda)

        # Phase 4: Layer-wise Calibration (optimize SVD factors to minimize reconstruction error)
        if calibration_steps > 0:
            try:
                if use_als:
                    print(f"Starting Phase 4 ALS calibration ({als_iters} iterations)...")
                    self.phase4_als_calibration(calib_loader, num_iters=als_iters,
                                                 update_sigma=True, token_sample_ratio=token_sample_ratio)
                else:
                    print(f"Starting Phase 4 M-optimization calibration...")
            except Exception as e:
                print(f"  Warning: Phase 4 calibration failed ({type(e).__name__}: {e}), skipping...")
                print("  Proceeding without calibration.")
                import traceback
                traceback.print_exc()
        else:
            print("Phase 4: Skipped (calibration_steps=0)")

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
                                update_sigma: bool = True, token_sample_ratio: float = 0.2) -> None:
        """
        Phase 4: ALS (Alternating Least Squares) Calibration.

        Unlike the M-optimization approach, ALS iteratively updates U and V separately:
        - Step A: Fix V, Σ, solve for U using least squares
        - Step B: Fix U, Σ, solve for V using least squares (FIXED: no U orthogonality assumption)
        - Step C (optional): Fix U, V, solve for diagonal scaling D (FIXED: r×r linear system)

        After each projection is calibrated, we write the updated SVD components back to
        the layer so that subsequent layers see the calibrated outputs.

        Mathematical formulation:
        Given W' = U @ Σ @ V^T, we want to minimize ||X @ W'^T - X @ W^T||_F^2

        Step A: Fix V, Σ, solve U
            Z = X @ V @ Σ (N × r)
            U^T = (Z^T Z)^{-1} Z^T Y → U = (solution)^T

        Step B: Fix U, Σ, solve V (CORRECTED - no U orthogonality assumption)
            Let U_s = U * S (out_dim × r)
            Z = Y @ U_s @ (U_s^T U_s)^{-1}  (target for X @ V)
            V = lstsq(X, Z)

        Step C: Fix U, V, solve D (r×r linear system)
            A = X @ V, B = U
            h = (A * (Y @ B)).sum(dim=0)
            G = (A^T A) ⊙ (B^T B)  (Hadamard product)
            d = solve(G, h)

        Args:
            calib_loader: Calibration data loader
            num_iters: Number of ALS iterations (default: 2)
            update_sigma: Whether to update diagonal scaling in Step C (default: True)
            token_sample_ratio: Ratio of tokens to sample per sequence to avoid OOM (default: 0.1)
        """
        print(f"Phase 4: ALS Calibration ({num_iters} iterations, update_sigma={update_sigma}, token_sample={token_sample_ratio:.0%})...")

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

                    # ALS iterations
                    for als_iter in range(num_iters):
                        # Step A: Fix V, S, solve U (with regularization)
                        Z = (X @ V) * S  # (N, r)
                        # Use lstsq for better numerical stability
                        U_T_new = torch.linalg.lstsq(Z, Y).solution  # (r, out_dim)
                        U = U_T_new.T  # (out_dim, r)
                        del Z, U_T_new

                        # Step B: Fix U, S, solve V (with regularization)
                        U_s = U * S  # (out_dim, r)
                        G = U_s.T @ U_s + reg * torch.eye(rank, device=self.device)
                        Z_target = (Y @ U_s) @ torch.linalg.inv(G)  # (N, r)
                        # Z_target = torch.linalg.solve(G, (Y @ U_s).T).T
                        # Solve X @ V = Z_target -> V = lstsq(X, Z_target)
                        V = torch.linalg.lstsq(X, Z_target).solution  # (in_dim, r)
                        del U_s, G, Z_target

                    # Step C: Fix U, V, solve D                    
                    if update_sigma:
                        A = X @ V  # (N, r)
                        YB = Y @ U  # (N, r)

                        h = (A * YB).sum(dim=0)  # (r,)
                        AtA = A.T @ A  # (r, r)
                        BtB = U.T @ U  # (r, r)
                        G = AtA * BtB  # Hadamard product (r, r)

                        d = torch.linalg.solve(G + reg * torch.eye(rank, device=self.device), h)
                        S = torch.abs(d)  # Keep positive
                        del A, YB, h, AtA, BtB, G, d

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
