"""
Optimization utilities for NVIDIA GPUs and Triton.
This module provides optimized implementations for NVIDIA hardware acceleration.
"""

import torch
import torch.utils.cpp_extension  # Import CUDA headers first
import warnings
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Union, Dict, Any, Tuple
import math
import logging

# Try to import Triton
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    warnings.warn("Triton not available - using standard CUDA operations")

# Try to import Flash Attention
try:
    import flash_attn
    from flash_attn.flash_attention import FlashAttention
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    warnings.warn("Flash Attention not available - using standard attention")

# Try to import XFormers
try:
    import xformers
    from xformers.ops import memory_efficient_attention
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("XFormers not available - using standard attention")

class OptimizedLinear(nn.Module):
    """Optimized linear layer for NVIDIA GPUs."""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        # Use Kaiming initialization for better training stability
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        # Use PyTorch's optimized linear operation
        return F.linear(x, self.weight, self.bias)

def optimize_for_nvidia(model: Optional[nn.Module] = None):
    """Enable NVIDIA-specific optimizations."""
    # Enable TF32 precision for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    
    # Enable gradient checkpointing for memory efficiency
    torch.backends.cudnn.deterministic = False
    
    if model is not None and FLASH_ATTN_AVAILABLE:
        # Replace standard attention with flash attention where possible
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                # Get the module parameters
                embed_dim = module.embed_dim
                num_heads = module.num_heads
                batch_first = module.batch_first
                
                # Create flash attention instance
                flash_attn = FlashAttention(
                    softmax_scale=1.0 / math.sqrt(embed_dim // num_heads),
                    attention_dropout=0.0
                )
                
                # Store the flash attention module
                module._flash_attn = flash_attn
                
                # Override the forward method to use flash attention
                def new_forward(self, query, key, value, *args, **kwargs):
                    # Reshape inputs for flash attention
                    B, L, E = query.shape
                    H = self.num_heads
                    D = E // H
                    
                    # Convert to bfloat16 for flash attention
                    query = query.view(B, L, H, D).to(torch.bfloat16)
                    key = key.view(B, L, H, D).to(torch.bfloat16)
                    value = value.view(B, L, H, D).to(torch.bfloat16)
                    
                    # Apply flash attention
                    output = self._flash_attn(query, key, value)
                    
                    # Convert back to original dtype and shape
                    output = output.to(query.dtype).view(B, L, E)
                    
                    return output, None
                
                # Bind the new forward method
                import types
                module.forward = types.MethodType(new_forward, module)

def get_nvidia_optimizer(params, lr=1e-3, weight_decay=0.01):
    """Get an optimizer configured for NVIDIA GPUs."""
    return AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
