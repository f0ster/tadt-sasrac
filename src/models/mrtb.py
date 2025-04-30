import torch
import torch.nn as nn
from src.utils.optimization import OptimizedLinear
from flash_attn.flash_attn_interface import flash_attn_func

class MRTB(nn.Module):
    """Multi-Range Transformer Block."""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.n_heads = 8
        self.head_dim = d_model // self.n_heads
        assert d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        
        # QKV projections
        self.qkv = OptimizedLinear(d_model, 3 * d_model)
        
        # Output projection
        self.out_proj = OptimizedLinear(d_model, d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            OptimizedLinear(d_model, d_model * 4),
            nn.ReLU(),
            OptimizedLinear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, route=None):
        B, L, D = x.shape
        
        # Layer norm before attention
        x_norm = self.norm1(x)
        
        # Project to q, k, v
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # Convert to bfloat16 for flash attention
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        
        # Flash attention
        attn_output = flash_attn_func(q, k, v)
        
        # Convert back to original dtype
        attn_output = attn_output.to(x.dtype)
        
        # Reshape and project output
        attn_output = attn_output.reshape(B, L, D)
        attn_output = self.out_proj(attn_output)
        
        # First residual connection
        x = x + attn_output
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x