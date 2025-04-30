import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import json
import thop
from torch.profiler import profile, record_function, ProfilerActivity
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd
from typing import Any

def analyze_model_parameters(model: nn.Module) -> Dict[str, Union[int, float]]:
    """Analyze model parameters and compute statistics.
    
    Args:
        model (nn.Module): Model to analyze
        
    Returns:
        Dict[str, Union[int, float]]: Dictionary containing parameter statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Analyze by layer type
    layer_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_stats[name] = {
                "type": "Linear",
                "params": sum(p.numel() for p in module.parameters()),
                "input_dim": module.in_features,
                "output_dim": module.out_features
            }
        elif isinstance(module, nn.Embedding):
            layer_stats[name] = {
                "type": "Embedding",
                "params": sum(p.numel() for p in module.parameters()),
                "num_embeddings": module.num_embeddings,
                "embedding_dim": module.embedding_dim
            }
        elif isinstance(module, nn.LayerNorm):
            layer_stats[name] = {
                "type": "LayerNorm",
                "params": sum(p.numel() for p in module.parameters()),
                "normalized_shape": module.normalized_shape
            }
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "layer_stats": layer_stats
    }

def analyze_model_complexity(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cuda"
) -> Dict[str, Union[int, float]]:
    """Analyze model complexity using FLOPs and MACs.
    
    Args:
        model (nn.Module): Model to analyze
        input_shape (Tuple[int, ...]): Shape of input tensor
        device (str): Device to run analysis on
        
    Returns:
        Dict[str, Union[int, float]]: Dictionary containing complexity metrics
    """
    model = model.to(device)
    dummy_input = torch.randn(input_shape, device=device)
    
    # Compute FLOPs and MACs
    macs, params = thop.profile(model, inputs=(dummy_input,))
    flops = 2 * macs  # Assuming each MAC operation requires 2 FLOPs
    
    return {
        "flops": flops,
        "macs": macs,
        "params": params
    }

def analyze_training_speed(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 32,
    num_iterations: int = 100,
    device: str = "cuda"
) -> Dict[str, float]:
    """Analyze model training speed.
    
    Args:
        model (nn.Module): Model to analyze
        input_shape (Tuple[int, ...]): Shape of input tensor
        batch_size (int): Batch size for analysis
        num_iterations (int): Number of iterations to run
        device (str): Device to run analysis on
        
    Returns:
        Dict[str, float]: Dictionary containing speed metrics
    """
    model = model.to(device)
    model.train()
    
    # Create dummy input and target
    dummy_input = torch.randn((batch_size, *input_shape), device=device)
    dummy_target = torch.randint(0, 100, (batch_size,), device=device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters())
    
    # Warmup
    for _ in range(10):
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = torch.nn.functional.cross_entropy(output, dummy_target)
        loss.backward()
        optimizer.step()
    
    # Measure speed
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = torch.nn.functional.cross_entropy(output, dummy_target)
        loss.backward()
        optimizer.step()
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
    
    return {
        "iterations_per_second": num_iterations / elapsed_time,
        "seconds_per_iteration": elapsed_time / num_iterations,
        "total_time": elapsed_time
    }

def analyze_memory_usage(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 32,
    device: str = "cuda"
) -> Dict[str, int]:
    """Analyze model memory usage.
    
    Args:
        model (nn.Module): Model to analyze
        input_shape (Tuple[int, ...]): Shape of input tensor
        batch_size (int): Batch size for analysis
        device (str): Device to run analysis on
        
    Returns:
        Dict[str, int]: Dictionary containing memory usage metrics
    """
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn((batch_size, *input_shape), device=device)
    
    # Measure memory before forward pass
    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated()
    
    # Forward pass
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Measure memory after forward pass
    peak_memory = torch.cuda.max_memory_allocated()
    end_memory = torch.cuda.memory_allocated()
    
    return {
        "start_memory": start_memory,
        "peak_memory": peak_memory,
        "end_memory": end_memory,
        "memory_increase": end_memory - start_memory
    }

def analyze_expert_usage(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 32,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """Analyze expert usage in MoE layers.
    
    Args:
        model (nn.Module): Model to analyze
        input_shape (Tuple[int, ...]): Shape of input tensor
        batch_size (int): Batch size for analysis
        device (str): Device to run analysis on
        
    Returns:
        Dict[str, torch.Tensor]: Dictionary containing expert usage statistics
    """
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn((batch_size, *input_shape), device=device)
    
    # Collect gate values
    gate_values = {}
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                gate_values[name] = output[1]  # Assuming gate values are second output
            else:
                gate_values[name] = output
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, MoE):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return gate_values

def analyze_recommendation_quality(
    recommendations: torch.Tensor,
    ground_truth: torch.Tensor,
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """Analyze recommendation quality using various metrics.
    
    Args:
        recommendations (torch.Tensor): Tensor of recommended items
        ground_truth (torch.Tensor): Tensor of ground truth items
        k_values (List[int]): List of k values for metrics
        
    Returns:
        Dict[str, float]: Dictionary containing quality metrics
    """
    metrics = {}
    
    for k in k_values:
        # Hit Rate
        hits = (recommendations[:, :k] == ground_truth.unsqueeze(1)).any(dim=1)
        hit_rate = hits.float().mean().item()
        metrics[f"hit_rate@{k}"] = hit_rate
        
        # NDCG
        relevance = (recommendations[:, :k] == ground_truth.unsqueeze(1)).float()
        dcg = (relevance / torch.log2(torch.arange(2, k + 2, device=relevance.device))).sum(dim=1)
        idcg = (1.0 / torch.log2(torch.arange(2, k + 1, device=relevance.device))).sum()
        ndcg = (dcg / idcg).mean().item()
        metrics[f"ndcg@{k}"] = ndcg
        
        # Recall
        recall = hits.float().mean().item()
        metrics[f"recall@{k}"] = recall
    
    return metrics

def save_analysis_results(
    results: Dict[str, Any],
    save_path: Union[str, Path]
) -> None:
    """Save analysis results to file.
    
    Args:
        results (Dict[str, Any]): Analysis results to save
        save_path (Union[str, Path]): Path to save results
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert tensors to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            serializable_results[key] = value.cpu().numpy().tolist()
        else:
            serializable_results[key] = value
    
    with open(save_path, "w") as f:
        json.dump(serializable_results, f, indent=4)

def analyze_training_speed(log_df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze training speed and efficiency.
    
    Args:
        log_df: DataFrame containing training logs
        
    Returns:
        Dictionary containing training statistics
    """
    total_time = log_df['time'].max() - log_df['time'].min()
    steps_per_second = len(log_df) / total_time if total_time > 0 else 0
    
    return {
        'total_time_seconds': total_time,
        'steps_per_second': steps_per_second,
        'average_loss': log_df['loss'].mean(),
        'final_loss': log_df['loss'].iloc[-1]
    }

def evaluate_model_performance(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate model performance on validation data.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing performance metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            cen, ctx, seqlen = [b.to(device) for b in batch]
            if hasattr(model, 'get_attention_weights'):
                pred, _ = model(cen.unsqueeze(1), seqlen)
            else:
                pred = model(cen.unsqueeze(1))[:,-1]
            
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(ctx.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, np.argmax(all_predictions, axis=1), average='weighted')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': confusion_matrix(all_targets, np.argmax(all_predictions, axis=1))
    } 