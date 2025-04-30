import torch
import sys

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Current CUDA device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    print("Device count:", torch.cuda.device_count())
    
    # Test CUDA operations
    x = torch.rand(5, 3).cuda()
    y = torch.rand(5, 3).cuda()
    z = x + y
    print("\nCUDA operation test successful!")
    print("Result shape:", z.shape)
    print("Result device:", z.device)
else:
    print("\nCUDA is not available!") 