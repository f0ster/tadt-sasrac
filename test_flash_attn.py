import torch
import flash_attn
from flash_attn.flash_attn_interface import flash_attn_func

def test_flash_attn():
    # Create test tensors
    batch_size = 2
    seq_len = 4
    n_heads = 2
    head_dim = 64
    
    # Use bf16 data type
    q = torch.randn(batch_size, seq_len, n_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(batch_size, seq_len, n_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(batch_size, seq_len, n_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    
    # Test flash attention
    try:
        output = flash_attn_func(q, k, v)
        print("Flash attention test successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output device: {output.device}")
        print(f"Output dtype: {output.dtype}")
        print(f"Flash attention version: {flash_attn.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        return True
    except Exception as e:
        print(f"Flash attention test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_flash_attn() 