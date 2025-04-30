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

## Features

- **Mixture of Experts**: Multiple specialized networks with dynamic gating
- **Multi-Resolution Processing**: Hierarchical sequence understanding
- **Dynamic Routing**: Task-aware information flow control
- **NVIDIA Optimizations**: TF32 precision, cuDNN benchmarking, and gradient checkpointing
- **Flash Attention Support**: Optional integration with Flash Attention for improved performance
- **Memory Efficiency**: Gradient checkpointing and optimized architecture
- **Comprehensive Analysis**: Built-in tools for model analysis and visualization
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
├── notebooks/            # Jupyter notebooks for analysis
├── src/                  # Source code
│   ├── train.py         # Training script
│   ├── evaluate.py      # Evaluation script
│   └── utils/           # Utility modules
│       ├── optimization.py  # NVIDIA optimizations
│       ├── analysis.py  # Analysis utilities
│       ├── visualization.py  # Visualization tools
│       ├── recommendation.py # Recommendation utilities
│       └── report.py    # Report generation
└── tests/               # Test suite
```

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
- Sets up MLflow logging
- Configures learning rate scheduling with warmup

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

## Report Generation

Generate comprehensive reports using the ReportGenerator:

```python
from src.utils.report import ReportGenerator

# Initialize report generator
report_generator = ReportGenerator("reports", model, "tadt_rec")

# Generate reports
model_report = report_generator.generate_model_report(input_shape)
training_report = report_generator.generate_training_report(train_metrics)
recommendation_report = report_generator.generate_recommendation_report(
    user_sequences, ground_truth
)

# Generate summary
summary_path = report_generator.generate_summary_report()
```

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
