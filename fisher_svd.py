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

        Memory-efficient implementation with:
        - Multi-GPU model parallelism (if available)
        - Gradient checkpointing
        - Per-sample gradient accumulation
        """

        print("  Using end-to-end task loss (cross-entropy) for Fisher estimation...")

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

        # Accumulate Fisher information
        num_samples = 0
        total_loss = 0.0

        for batch in tqdm(calib_loader):
            # Move batch to appropriate device (first GPU for multi-GPU, or single device)
            target_device = self.devices[0] if self.use_multi_gpu else self.device
            batch = {k: v.to(target_device) for k, v in batch.items()}

            # Zero gradients
            self.model.zero_grad()

            try:
                # Forward pass with cross-entropy loss
                outputs = self.model(**batch, labels=batch['input_ids'])
                loss = outputs.loss
                total_loss += loss.item()

                # Backward pass
                loss.backward()

                # Accumulate squared gradients (Fisher information)
                # Use stored SVD layer references instead of find_layers
                for layer_idx in self.svd_layer_refs:
                    for name, svd_layer in self.svd_layer_refs[layer_idx].items():
                        if svd_layer.sigma.grad is not None:
                            self.fisher_info[layer_idx][name] += svd_layer.sigma.grad.pow(2).cpu()

                num_samples += 1

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  Warning: OOM at sample {num_samples}, trying to recover...")
                    torch.cuda.empty_cache()
                    # Try with gradient accumulation fallback
                    continue
                else:
                    raise e

            # Clear cache after each batch
            if num_samples % 4 == 0:
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

        print(f"  Estimated Fisher information using {num_samples} samples")

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

    def compute_importance_scores(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute importance scores for all singular values.

        Enhanced scoring formula:
        Score_i = σ_i² × F_ii × layer_factor

        Where layer_factor increases for later layers (more important for generation).

        Returns:
            Dictionary of importance scores per layer and sublayer
        """
        importance_scores = {}
        fisher_used = 0
        fisher_fallback = 0

        num_layers = len(self.layers)

        for layer_idx in self.svd_components:
            layer_scores = {}

            # Layer position factor: later layers get higher weight
            # Using smooth sigmoid-like curve: factor ranges from 0.5 to 1.5
            # This protects later layers which are more important for generation quality
            layer_position = layer_idx / (num_layers - 1) if num_layers > 1 else 0.5
            layer_factor = 0.5 + layer_position  # Range: [0.5, 1.5]

            for name in self.svd_components[layer_idx]:
                U, S, VT, bias = self.svd_components[layer_idx][name]

                if layer_idx in self.fisher_info and name in self.fisher_info[layer_idx]:
                    F = self.fisher_info[layer_idx][name]
                    # Score_i = σ_i² × F_ii × layer_factor
                    # Add small epsilon to F to avoid all-zero scores
                    F_regularized = F + 1e-10
                    scores = S.pow(2) * F_regularized * layer_factor
                    fisher_used += 1
                else:
                    # Fallback to magnitude-based scoring with layer factor
                    scores = S.pow(2) * layer_factor
                    fisher_fallback += 1

                layer_scores[name] = scores
            importance_scores[layer_idx] = layer_scores

        print(f"  Importance scores: {fisher_used} with Fisher, {fisher_fallback} fallback to magnitude")
        return importance_scores

    def phase3_global_truncation(self, ratio: float, min_rank: int = 16) -> None:
        """
        Phase 3: Global truncation based on importance scores.

        Following the algorithm:
        1. Compute Score_i^(l) = (σ_i^(l))² × F_σi^(l) for all singular values
        2. Normalize scores per layer for balanced truncation
        3. Flatten all scores into a list S and sort globally (descending)
        4. Keep top ρ proportion of scores, respecting minimum rank constraints
        5. Reconstruct W'^(l) = U^(l) Σ'^(l) V^(l)^T

        Args:
            ratio: Target retention ratio (0-1). Higher means more parameters kept.
            min_rank: Minimum rank to keep per layer (default: 16)
        """
        print(f"Phase 3: Global Truncation (target ratio: {ratio:.2%}, min_rank: {min_rank})...")

        # Compute importance scores: Score_i = σ_i² × F_ii
        importance_scores = self.compute_importance_scores()

        # Layer-wise normalization for balanced truncation
        # This prevents some layers from dominating the global selection
        normalized_scores = {}
        for layer_idx in importance_scores:
            normalized_scores[layer_idx] = {}
            for name in importance_scores[layer_idx]:
                scores = importance_scores[layer_idx][name]
                # Normalize by layer's total importance (L2 norm)
                layer_norm = torch.norm(scores).item() + 1e-10
                normalized = scores / layer_norm
                normalized_scores[layer_idx][name] = normalized

        # Collect all scores with their identifiers (layer_idx, name, singular_value_idx)
        # Use normalized scores for ranking but store original scores for debugging
        all_scores = []
        for layer_idx in normalized_scores:
            for name in normalized_scores[layer_idx]:
                scores = normalized_scores[layer_idx][name]
                original_scores = importance_scores[layer_idx][name]
                for i, (norm_score, orig_score) in enumerate(zip(scores, original_scores)):
                    all_scores.append((norm_score.item(), layer_idx, name, i, orig_score.item()))

        # Sort by normalized importance (descending)
        all_scores.sort(key=lambda x: x[0], reverse=True)

        # Calculate total singular values and target count
        total_sv_count = len(all_scores)

        # Calculate per-layer parameter constraints for proper compression ratio
        # For W ∈ R^{m×n} with rank r: params = r(m+n), original = mn
        # To achieve compression ratio ρ: r(m+n) ≈ ρ × mn → r ≈ ρmn/(m+n)
        layer_max_rank = {}
        layer_min_rank = {}
        total_original_params = 0
        for layer_idx in self.svd_components:
            for name in self.svd_components[layer_idx]:
                U, S, VT, _ = self.svd_components[layer_idx][name]
                m, n = U.shape[0], VT.shape[1]
                total_original_params += m * n
                original_rank = len(S)

                # Maximum rank that satisfies compression ratio
                max_rank = int(m * n * ratio / (m + n))
                max_rank = max(1, min(max_rank, original_rank))

                # Minimum rank constraint: at least min_rank or 10% of original, whichever is smaller
                min_r = min(min_rank, max(1, int(original_rank * 0.1)))
                min_r = min(min_r, max_rank)  # Don't exceed max_rank

                layer_max_rank[(layer_idx, name)] = max_rank
                layer_min_rank[(layer_idx, name)] = min_r

        # First, allocate minimum ranks for all layers
        kept_indices: Dict[int, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
        kept_count = 0

        # Pre-allocate minimum ranks by taking top singular values per layer
        for layer_idx in self.svd_components:
            for name in self.svd_components[layer_idx]:
                min_r = layer_min_rank[(layer_idx, name)]
                # Take top min_r singular values for this layer
                for i in range(min_r):
                    kept_indices[layer_idx][name].add(i)
                    kept_count += 1

        # Calculate remaining budget after minimum allocation
        total_target_sv = sum(layer_max_rank.values())
        remaining_budget = total_target_sv - kept_count

        print(f"  Pre-allocated {kept_count} singular values for minimum ranks")
        print(f"  Remaining budget: {remaining_budget}")

        # Fill remaining budget using global selection
        if remaining_budget > 0:
            for norm_score, layer_idx, name, idx, orig_score in all_scores:
                if remaining_budget <= 0:
                    break

                # Skip if already kept (from minimum allocation)
                if idx in kept_indices[layer_idx][name]:
                    continue

                current_kept = len(kept_indices[layer_idx][name])
                max_for_layer = layer_max_rank.get((layer_idx, name), 0)

                # Check if we haven't exceeded max rank for this layer
                if current_kept < max_for_layer:
                    kept_indices[layer_idx][name].add(idx)
                    kept_count += 1
                    remaining_budget -= 1

        # Truncate SVD components
        truncation_samples = []  # For debug output
        rank_stats = []  # Track min/max ranks
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
                 calibration_steps: int = 50) -> nn.Module:
        """
        Full compression pipeline.

        Args:
            calib_loader: Calibration data loader
            ratio: Target compression ratio (0-1)
            whitening_mat: Optional whitening matrices from SVD-LLM
            use_low_resource: Use memory-efficient proxy loss (default: False, use true CE loss)
            calibration_steps: Number of calibration steps per layer (default: 50)

        Returns:
            Compressed model
        """
        # Phase 1: SVD Decomposition
        self.phase1_svd_decomposition(whitening_mat)

        # Phase 2: Sensitivity Estimation
        self.phase2_sensitivity_estimation(calib_loader, use_low_resource)

        # Phase 3: Global Truncation
        self.phase3_global_truncation(ratio)

        # Phase 4: Layer-wise Calibration (optimize SVD factors to minimize reconstruction error)
        self.phase4_calibration(calib_loader, calibration_steps)

        # Apply compression to model
        self.apply_compression(ratio)

        return self.model

    def phase4_calibration(self, calib_loader: List[Dict], num_steps: int = 50) -> None:
        """
        Phase 4: Layer-wise calibration to fine-tune SVD factors.

        After truncation, optimize the remaining SVD factors to minimize
        reconstruction error on calibration data. This is similar to the
        calibration step in GPTQ and other advanced compression methods.

        Args:
            calib_loader: Calibration data loader
            num_steps: Number of optimization steps per layer
        """
        print(f"Phase 4: Layer-wise Calibration ({num_steps} steps per layer)...")

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

        outs = torch.zeros_like(inps)
        total_loss_before = 0.0
        total_loss_after = 0.0

        for layer_idx in tqdm(range(len(self.layers))):
            layer = self.layers[layer_idx].to(self.device)
            subset = find_layers(layer)

            if layer_idx not in self.svd_components:
                # Just forward through this layer
                with torch.no_grad():
                    for j in range(inps.shape[0]):
                        if position_ids is not None and "opt" not in self.model_name:
                            outs[j] = layer(inps[j].unsqueeze(0),
                                           attention_mask=attention_masks[j].unsqueeze(0),
                                           position_ids=position_ids[j].unsqueeze(0))[0]
                        else:
                            outs[j] = layer(inps[j].unsqueeze(0),
                                           attention_mask=attention_masks[j].unsqueeze(0))[0]
                self.layers[layer_idx] = layer.cpu()
                inps = outs.clone()
                torch.cuda.empty_cache()
                continue

            # Capture original outputs for this layer
            original_outs = torch.zeros_like(inps)
            with torch.no_grad():
                for j in range(inps.shape[0]):
                    if position_ids is not None and "opt" not in self.model_name:
                        original_outs[j] = layer(inps[j].unsqueeze(0),
                                                  attention_mask=attention_masks[j].unsqueeze(0),
                                                  position_ids=position_ids[j].unsqueeze(0))[0]
                    else:
                        original_outs[j] = layer(inps[j].unsqueeze(0),
                                                  attention_mask=attention_masks[j].unsqueeze(0))[0]

            # Create trainable SVD layers for calibration
            svd_layers = {}
            for name in subset:
                if name not in self.svd_components[layer_idx]:
                    continue

                U, S, VT, bias = self.svd_components[layer_idx][name]
                rank = len(S)
                out_features, in_features = U.shape[0], VT.shape[1]

                # Create trainable linear layers
                sqrt_sigma = torch.sqrt(S)
                svd_u = (U * sqrt_sigma).to(dtype).to(self.device)
                svd_v = (sqrt_sigma.unsqueeze(1) * VT).to(dtype).to(self.device)

                u_proj = nn.Linear(rank, out_features, bias=(bias is not None)).to(self.device)
                v_proj = nn.Linear(in_features, rank, bias=False).to(self.device)

                u_proj.weight.data = svd_u
                v_proj.weight.data = svd_v
                if bias is not None:
                    u_proj.bias.data = bias.to(dtype).to(self.device)

                # Make weights trainable
                u_proj.weight.requires_grad = True
                v_proj.weight.requires_grad = True

                svd_layers[name] = (u_proj, v_proj, bias is not None)

                # Replace in layer temporarily
                svd_linear = SVDLinear(v_proj, u_proj)
                self._set_module_by_name(layer, name, svd_linear)

            # Optimize SVD layers
            all_params = []
            for name, (u_proj, v_proj, _) in svd_layers.items():
                all_params.extend([u_proj.weight, v_proj.weight])

            if len(all_params) > 0:
                optimizer = torch.optim.Adam(all_params, lr=1e-4)

                for step in range(num_steps):
                    total_loss = 0.0
                    for j in range(inps.shape[0]):
                        optimizer.zero_grad()

                        if position_ids is not None and "opt" not in self.model_name:
                            out_j = layer(inps[j].unsqueeze(0),
                                         attention_mask=attention_masks[j].unsqueeze(0),
                                         position_ids=position_ids[j].unsqueeze(0))[0]
                        else:
                            out_j = layer(inps[j].unsqueeze(0),
                                         attention_mask=attention_masks[j].unsqueeze(0))[0]

                        # Reconstruction loss
                        loss = ((out_j.float() - original_outs[j].unsqueeze(0).float()) ** 2).mean()
                        total_loss += loss.item()

                        loss.backward()
                        optimizer.step()

                    if step == 0:
                        total_loss_before += total_loss / inps.shape[0]

                total_loss_after += total_loss / inps.shape[0]

                # Update SVD components with calibrated weights
                for name, (u_proj, v_proj, has_bias) in svd_layers.items():
                    # Convert back to U, S, VT format
                    # W = u_proj.weight @ v_proj.weight = U @ sqrt(S) @ sqrt(S) @ VT = U @ S @ VT
                    with torch.no_grad():
                        W_new = u_proj.weight.data @ v_proj.weight.data
                        U_new, S_new, VT_new = torch.linalg.svd(W_new.float(), full_matrices=False)

                        # Keep only the truncated rank
                        rank = v_proj.weight.shape[0]
                        U_new = U_new[:, :rank]
                        S_new = S_new[:rank]
                        VT_new = VT_new[:rank, :]

                        bias_data = u_proj.bias.data.cpu() if has_bias else None
                        self.svd_components[layer_idx][name] = (U_new.cpu(), S_new.cpu(), VT_new.cpu(), bias_data)

            # Forward through calibrated layer for next layer's input
            with torch.no_grad():
                for j in range(inps.shape[0]):
                    if position_ids is not None and "opt" not in self.model_name:
                        outs[j] = layer(inps[j].unsqueeze(0),
                                       attention_mask=attention_masks[j].unsqueeze(0),
                                       position_ids=position_ids[j].unsqueeze(0))[0]
                    else:
                        outs[j] = layer(inps[j].unsqueeze(0),
                                       attention_mask=attention_masks[j].unsqueeze(0))[0]

            self.layers[layer_idx] = layer.cpu()
            inps = outs.clone()
            torch.cuda.empty_cache()

        avg_loss_before = total_loss_before / len(self.layers)
        avg_loss_after = total_loss_after / len(self.layers)
        print(f"  Reconstruction loss: {avg_loss_before:.6f} -> {avg_loss_after:.6f}")
        print(f"  Improvement: {(1 - avg_loss_after / avg_loss_before) * 100:.1f}%")


def fisher_aware_svd_compression(model_name: str, model: nn.Module,
                                  calib_loader: List[Dict], ratio: float,
                                  whitening_mat: Optional[Dict] = None,
                                  device: str = "cuda",
                                  use_low_resource: bool = False,
                                  num_gpus: int = 1,
                                  calibration_steps: int = 50) -> nn.Module:
    """
    Main entry point for Fisher-Aware SVD compression.

    Args:
        model_name: Name of the model (e.g., "llama", "mistral", "opt")
        model: The model to compress
        calib_loader: Calibration data loader
        ratio: Target compression ratio (0-1). Higher means more parameters kept.
        whitening_mat: Optional whitening matrices from SVD-LLM profiling
        device: Device to use for computation
        use_low_resource: Use memory-efficient proxy loss (default: False, use true CE loss)
        num_gpus: Number of GPUs to use for model parallelism (default: 1)

    Returns:
        Compressed model
    """
    print(f"Fisher-Aware SVD Compression")
    print(f"  Mode: {'Proxy Loss (low resource)' if use_low_resource else 'Cross-Entropy Loss (full)'}")
    print(f"  GPUs: {num_gpus}")

    compressor = FisherAwareSVD(model, model_name, device, num_gpus=num_gpus)
    return compressor.compress(calib_loader, ratio, whitening_mat, use_low_resource, calibration_steps)


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
