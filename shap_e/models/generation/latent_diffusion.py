from typing import Any, Dict

import torch
import torch.nn as nn


class SplitVectorDiffusion(nn.Module):
    def __init__(self, *, device: torch.device, wrapped: nn.Module, n_ctx: int, d_latent: int):
        super().__init__()
        self.device = device
        self.n_ctx = n_ctx
        self.d_latent = d_latent
        self.wrapped = wrapped

        if hasattr(self.wrapped, "cached_model_kwargs"):
            self.cached_model_kwargs = self.wrapped.cached_model_kwargs

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        h = x.reshape(x.shape[0], self.n_ctx, -1).permute(0, 2, 1)
        pre_channels = h.shape[1]
        h = self.wrapped(h, t, **kwargs)
        assert (
            h.shape[1] == pre_channels * 2
        ), "expected twice as many outputs for variance prediction"
        eps, var = torch.chunk(h, 2, dim=1)
        return torch.cat(
            [
                eps.permute(0, 2, 1).flatten(1),
                var.permute(0, 2, 1).flatten(1),
            ],
            dim=1,
        )
        
    def print_parameter_status(self):
        for name, param in self.wrapped.named_parameters():
            print(f"name: {name}, shape: {param.shape}, req grad: {param.requires_grad}")

    def freeze_all_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze_transformer_backbone(self, num_layers=None, reverse=False):
        total_num_layers = self.wrapped.backbone.layers
        if num_layers is None:
            num_layers = total_num_layers
        for name, param in self.wrapped.backbone.named_parameters():
            split_name = name.split('.')
            layer_num = int(split_name[1])
            if reverse:
                layer_num = total_num_layers - layer_num
            if layer_num >= num_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
                