import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import networkx as nx
from torch import nn
import torchviz
from pathlib import Path

def plot_training_metrics(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Metrics"
) -> None:
    """Plot training metrics over time.
    
    Args:
        metrics (Dict[str, List[float]]): Dictionary of metric names to values
        save_path (Optional[str]): Path to save the plot
        title (str): Plot title
    """
    plt.figure(figsize=(12, 6))
    for name, values in metrics.items():
        plt.plot(values, label=name)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_model_architecture(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    save_path: Optional[str] = None
) -> None:
    """Visualize model architecture using torchviz.
    
    Args:
        model (nn.Module): Model to visualize
        input_shape (Tuple[int, ...]): Shape of input tensor
        save_path (Optional[str]): Path to save the visualization
    """
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Generate graph
    dot = torchviz.make_dot(
        model(dummy_input),
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=True
    )
    
    if save_path:
        dot.render(save_path, format="png")
    
    return dot

def plot_attention_patterns(
    attention_weights: torch.Tensor,
    title: str = "Attention Patterns",
    save_path: Optional[str] = None
) -> None:
    """Plot attention patterns from transformer layers.
    
    Args:
        attention_weights (torch.Tensor): Attention weights tensor
        title (str): Plot title
        save_path (Optional[str]): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights.cpu().numpy(),
        cmap="viridis",
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_expert_usage(
    gate_values: torch.Tensor,
    num_experts: int,
    title: str = "Expert Usage Distribution",
    save_path: Optional[str] = None
) -> None:
    """Plot distribution of expert usage in MoE layer.
    
    Args:
        gate_values (torch.Tensor): Gate values from MoE layer
        num_experts (int): Number of experts
        title (str): Plot title
        save_path (Optional[str]): Path to save the plot
    """
    expert_usage = gate_values.sum(dim=0).cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_experts), expert_usage)
    plt.xlabel("Expert Index")
    plt.ylabel("Usage Count")
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_recommendation_distribution(
    recommendations: torch.Tensor,
    item_frequencies: Optional[torch.Tensor] = None,
    title: str = "Recommendation Distribution",
    save_path: Optional[str] = None
) -> None:
    """Plot distribution of recommended items.
    
    Args:
        recommendations (torch.Tensor): Tensor of recommended items
        item_frequencies (Optional[torch.Tensor]): Frequency of items in training data
        title (str): Plot title
        save_path (Optional[str]): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot recommendation distribution
    rec_counts = torch.bincount(recommendations.flatten()).cpu().numpy()
    plt.bar(range(len(rec_counts)), rec_counts, alpha=0.7, label="Recommendations")
    
    # Plot item frequencies if provided
    if item_frequencies is not None:
        plt.plot(
            item_frequencies.cpu().numpy(),
            color="red",
            alpha=0.5,
            label="Training Distribution"
        )
    
    plt.xlabel("Item ID")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None
) -> None:
    """Plot learning curves for training and validation.
    
    Args:
        train_losses (List[float]): Training losses
        val_losses (List[float]): Validation losses
        train_metrics (Optional[Dict[str, List[float]]]): Additional training metrics
        val_metrics (Optional[Dict[str, List[float]]]): Additional validation metrics
        save_path (Optional[str]): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # Plot losses
    ax1.plot(train_losses, label="Training Loss")
    ax1.plot(val_losses, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True)
    
    # Plot additional metrics if provided
    if train_metrics and val_metrics:
        for name, values in train_metrics.items():
            ax2.plot(values, label=f"Train {name}")
        for name, values in val_metrics.items():
            ax2.plot(values, label=f"Val {name}")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Metric Value")
        ax2.set_title("Metric Curves")
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_memory_usage(
    memory_stats: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """Plot GPU memory usage over time.
    
    Args:
        memory_stats (Dict[str, List[float]]): Dictionary of memory statistics
        save_path (Optional[str]): Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    for name, values in memory_stats.items():
        plt.plot(values, label=name)
    plt.xlabel("Step")
    plt.ylabel("Memory (MB)")
    plt.title("GPU Memory Usage")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show() 