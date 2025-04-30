# TADT-SASRec: Task-Aware Dynamic Transformer for Sequential Recommendation
*A next-generation sequential recommender that extends FAME design, with multi-objective routing and conditional compute.*

## Overview

A state-of-the-art sequential recommendation system that combines Mixture of Experts with Multi-Resolution Transformer Blocks for improved recommendation quality.

TADT-SASRec is a novel sequential recommendation model that addresses the limitations of existing approaches by:

- Implementing a Mixture of Experts architecture for specialized processing
- Using Multi-Resolution Transformer Blocks for hierarchical sequence understanding
- Incorporating dynamic routing through a Task-Aware Routing Controller
- Providing efficient training with NVIDIA optimizations and Flash Attention support
- Supporting mixed precision training with bfloat16 for improved performance
- Comprehensive experiment tracking with MLflow
- Custom CUDA kernels for optimized computation

## Features

- **Mixture of Experts**: Multiple specialized networks with dynamic gating
- **Multi-Resolution Processing**: Hierarchical sequence understanding
- **Dynamic Routing**: Task-aware information flow control
- **NVIDIA Optimizations**: TF32 precision, cuDNN benchmarking, and gradient checkpointing
- **Flash Attention Support**: Optional integration with Flash Attention for improved performance
- **Custom CUDA Kernels**: Optimized implementations for key operations
- **Memory Efficiency**: Gradient checkpointing and optimized architecture
- **MLflow Integration**: Complete experiment tracking and visualization
- **Modular Design**: Easy to extend and customize for different use cases

## Project Structure

```
tadt-sasrec/
├── configs/               # Configuration files
├── data/                  # Data processing and loading
├── models/               # Model implementations
│   ├── tadt_rec.py      # Main TADTRec model
│   ├── mrtb.py          # Multi-Resolution Transformer Block
│   └── tarc.py          # Task-Aware Routing Controller
├── kernels/             # Custom CUDA kernels
│   ├── attention.cu     # Optimized attention kernels
│   ├── routing.cu       # Dynamic routing kernels
│   └── expert.cu        # Expert selection kernels
├── notebooks/            # Jupyter notebooks for analysis
├── src/                  # Source code
│   ├── train.py         # Training script with MLflow integration
│   ├── evaluate.py      # Evaluation script
│   └── utils/           # Utility modules
│       ├── optimization.py  # NVIDIA optimizations
│       ├── analysis.py  # Analysis utilities
│       ├── visualization.py  # Visualization tools
│       └── recommendation.py # Recommendation utilities
└── tests/               # Test suite
```

## Custom Kernels and Optimizations

### 1. Optimized Attention Kernels

```ascii
┌─────────────────────────────────────────────────────────┐
│                     Input Sequence                       │
└───────────────────────────────┬─────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────┐
│                    Flash Attention                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Query     │    │    Key      │    │   Value     │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
│         │                  │                   │         │
│         v                  v                   v         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                 Attention Matrix                     │ │
│  └──────────────────────────┬──────────────────────────┘ │
│                             │                            │
│                             v                            │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                 Output Projection                    │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

Key optimizations:
- Block-wise computation for memory efficiency
- Shared memory caching for frequently accessed data
- Warp-level parallelism for matrix operations
- Fused operations to reduce memory bandwidth

### 2. Dynamic Routing Kernels

```ascii
┌─────────────────────────────────────────────────────────┐
│                     Input Features                       │
└───────────────────────────────┬─────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────┐
│                    Routing Controller                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  Feature    │    │  Attention  │    │   Gating    │  │
│  │  Projection │    │   Weights   │    │   Network   │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
│         │                  │                   │         │
│         v                  v                   v         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                 Expert Selection                     │ │
│  └──────────────────────────┬──────────────────────────┘ │
│                             │                            │
│                             v                            │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                 Output Routing                       │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

Optimizations:
- Parallel expert selection using warp-level primitives
- Efficient top-k selection with shared memory
- Fused operations for routing computation
- Memory-efficient expert representation

### 3. Performance Optimizations

1. **Memory Access Patterns**:
```python
# Coalesced memory access
__global__ void optimized_kernel(float* input, float* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < N; i += stride) {
        // Coalesced memory access
        output[i] = input[i] * 2.0f;
    }
}
```

2. **Shared Memory Usage**:
```python
# Shared memory caching
__global__ void shared_memory_kernel(float* input, float* output) {
    __shared__ float shared_data[BLOCK_SIZE];
    int tid = threadIdx.x;
    
    // Load data into shared memory
    shared_data[tid] = input[blockIdx.x * blockDim.x + tid];
    __syncthreads();
    
    // Process data from shared memory
    output[blockIdx.x * blockDim.x + tid] = shared_data[tid] * 2.0f;
}
```

3. **Warp-Level Operations**:
```python
# Warp-level primitives
__global__ void warp_ops_kernel(float* input, float* output) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Warp-level reduction
    float val = input[tid];
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    if (lane_id == 0) {
        output[warp_id] = val;
    }
}
```

### 4. Mixed Precision Training

```ascii
┌─────────────────────────────────────────────────────────┐
│                     Forward Pass                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  bfloat16   │    │  bfloat16   │    │  bfloat16   │  │
│  │  Input      │    │  Weights    │    │  Output     │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
│         │                  │                   │         │
│         v                  v                   v         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                 Accumulation                         │ │
│  │                 (float32)                           │ │
│  └──────────────────────────┬──────────────────────────┘ │
│                             │                            │
│                             v                            │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                 Backward Pass                        │ │
│  │                 (float32)                           │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

Key features:
- bfloat16 for forward pass
- float32 for accumulation and backward pass
- Automatic mixed precision (AMP) support
- Gradient scaling for stability

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tadt-sasrec.git
cd tadt-sasrec
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install Flash Attention for improved performance:
```bash
TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-13.3" pip install flash-attn --no-build-isolation --no-cache-dir
```

## Usage

### Training

```bash
bash scripts/run_training.sh
```

The training script automatically:
- Sets up NVIDIA optimizations
- Configures mixed precision training
- Enables Flash Attention if available
- Sets up MLflow logging with:
  - Model parameters and hyperparameters
  - Training metrics and validation scores
  - System metrics (GPU utilization, memory usage)
  - Model checkpoints and artifacts
  - Learning rate schedules
  - Training configuration

### MLflow Tracking

The project uses MLflow for comprehensive experiment tracking:

1. **View Experiments**:
```bash
mlflow ui
```

2. **Tracked Metrics**:
- Training loss and validation metrics
- Model performance (Hit Rate, NDCG, Recall)
- System metrics (GPU utilization, memory)
- Learning rate schedules
- Training time and efficiency

3. **Artifacts**:
- Model checkpoints
- Training configurations
- Evaluation results
- Performance plots

4. **Parameters**:
- Model architecture
- Training hyperparameters
- Optimization settings
- Data preprocessing parameters

### Evaluation

```bash
python src/evaluate.py --config configs/tadt_rec_config.yaml
```

### Analysis

Use the provided notebooks for model analysis and visualization:

```bash
jupyter notebook notebooks/model_analysis.ipynb
```

## Model Architecture

The TADT-SASRec model implements a sophisticated architecture combining Mixture of Experts with Multi-Resolution Transformer Blocks:

### 1. Multi-Resolution Transformer Blocks (MRTB)
```python
class MRTB(nn.Module):
    def __init__(self, d_model):
        self.d_model = d_model
        self.n_heads = 8
        self.head_dim = d_model // self.n_heads
        self.qkv = OptimizedLinear(d_model, 3 * d_model)
        self.out_proj = OptimizedLinear(d_model, d_model)
        self.ffn = nn.Sequential(
            OptimizedLinear(d_model, d_model * 4),
            nn.ReLU(),
            OptimizedLinear(d_model * 4, d_model)
        )
```
- Processes sequences at different resolutions
- Uses Flash Attention when available
- Implements optimized linear layers for NVIDIA GPUs
- Supports bfloat16 for improved performance

### 2. Task-Aware Routing Controller (TARC)
- Dynamically routes information through the network
- Controls flow through MRTB blocks
- Adapts to different recommendation tasks
- Outputs routing coefficients for each block

### 3. Main TADTRec Model
```python
class TADTRec(nn.Module):
    def __init__(self, n_items, d=128, K=6, max_len=40):
        self.item_emb = nn.Embedding(n_items, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.mrtbs = nn.ModuleList([MRTB(d) for _ in range(K)])
        self.ctrl = TARC(d, K*4)
        self.ln = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, n_items)
```
- Combines MRTB blocks with dynamic routing
- Projects to vocabulary size for recommendation
- Supports variable-length sequences
- Implements NVIDIA optimizations

### Key Optimizations

1. **NVIDIA Optimizations**:
```python
def optimize_for_nvidia(model):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
```

2. **Flash Attention Integration**:
```python
if FLASH_ATTN_AVAILABLE:
    attn_output = flash_attn_func(q, k, v)
```

3. **Memory Efficiency**:
- Gradient checkpointing for large sequences
- Optimized expert selection (top-k)
- Efficient routing computation
- Mixed precision training with bfloat16

## Performance Characteristics

- **Memory Usage**: ~11.6MB model size
- **Training Speed**: ~90 iterations/second on NVIDIA GeForce RTX 4090
- **Model Size**: 2.9M parameters
- **Batch Processing**: Efficient handling of variable-length sequences
- **Mixed Precision**: bfloat16 support for improved performance
- **Flash Attention**: Optional integration for faster attention computation

## Analysis Tools

The project includes comprehensive analysis tools:

### Model Analysis
- Parameter statistics
- Computational complexity
- Memory usage
- Architecture visualization

### Training Analysis
- Learning curves
- Training metrics
- Performance profiling
- Resource utilization

### Recommendation Analysis
- Quality metrics (Hit Rate, NDCG, Recall)
- Diversity analysis
- Fairness evaluation
- Visualization tools

## Performance

- Training speed: ~90 iterations/second on NVIDIA GeForce RTX 4090
- Model size: 2.9M parameters
- Memory usage: ~11.6MB model size

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tadt-sasrec,
  title={TADT-SASRec: Task-Aware Dynamic Transformer for Sequential Recommendation},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```
