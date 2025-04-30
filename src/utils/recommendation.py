import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

def generate_recommendations(
    model: nn.Module,
    user_sequences: torch.Tensor,
    k: int = 10,
    device: str = "cuda",
    batch_size: int = 256
) -> torch.Tensor:
    """Generate recommendations for user sequences.
    
    Args:
        model (nn.Module): Trained recommendation model
        user_sequences (torch.Tensor): User sequences to generate recommendations for
        k (int): Number of recommendations to generate
        device (str): Device to run inference on
        batch_size (int): Batch size for inference
        
    Returns:
        torch.Tensor: Tensor of recommended items
    """
    model = model.to(device)
    model.eval()
    
    recommendations = []
    num_users = len(user_sequences)
    
    with torch.no_grad():
        for i in tqdm(range(0, num_users, batch_size), desc="Generating recommendations"):
            batch = user_sequences[i:i + batch_size].to(device)
            
            # Get model output
            if isinstance(model, TADTRec):
                output, _ = model(batch, torch.ones(len(batch), device=device))
            else:
                output = model(batch)[:,-1]
            
            # Get top-k recommendations
            _, top_k = torch.topk(output, k, dim=1)
            recommendations.append(top_k.cpu())
    
    return torch.cat(recommendations, dim=0)

def evaluate_recommendations(
    recommendations: torch.Tensor,
    ground_truth: torch.Tensor,
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """Evaluate recommendation quality using various metrics.
    
    Args:
        recommendations (torch.Tensor): Tensor of recommended items
        ground_truth (torch.Tensor): Tensor of ground truth items
        k_values (List[int]): List of k values for metrics
        
    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics
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
        
        # Precision
        precision = hits.float().mean().item()
        metrics[f"precision@{k}"] = precision
        
        # MRR
        ranks = (recommendations[:, :k] == ground_truth.unsqueeze(1)).nonzero()[:, 1] + 1
        mrr = (1.0 / ranks.float()).mean().item()
        metrics[f"mrr@{k}"] = mrr
    
    return metrics

def analyze_recommendation_diversity(
    recommendations: torch.Tensor,
    item_popularity: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Analyze recommendation diversity.
    
    Args:
        recommendations (torch.Tensor): Tensor of recommended items
        item_popularity (Optional[torch.Tensor]): Item popularity scores
        
    Returns:
        Dict[str, float]: Dictionary containing diversity metrics
    """
    metrics = {}
    
    # Coverage
    unique_items = torch.unique(recommendations)
    coverage = len(unique_items) / recommendations.size(1)
    metrics["coverage"] = coverage
    
    # Intra-list diversity
    if item_popularity is not None:
        # Gini coefficient
        sorted_pop = torch.sort(item_popularity[recommendations], dim=1)[0]
        n = sorted_pop.size(1)
        index = torch.arange(1, n + 1, device=sorted_pop.device)
        gini = ((2 * index - n - 1) * sorted_pop).sum(dim=1) / (n * sorted_pop.sum(dim=1))
        metrics["gini_coefficient"] = gini.mean().item()
    
    # Novelty
    if item_popularity is not None:
        novelty = -torch.log2(item_popularity[recommendations] + 1e-10).mean()
        metrics["novelty"] = novelty.item()
    
    return metrics

def analyze_recommendation_fairness(
    recommendations: torch.Tensor,
    item_groups: torch.Tensor,
    num_groups: int
) -> Dict[str, float]:
    """Analyze recommendation fairness across item groups.
    
    Args:
        recommendations (torch.Tensor): Tensor of recommended items
        item_groups (torch.Tensor): Group membership for each item
        num_groups (int): Number of groups
        
    Returns:
        Dict[str, float]: Dictionary containing fairness metrics
    """
    metrics = {}
    
    # Group representation
    group_counts = torch.zeros(num_groups, device=recommendations.device)
    for group in range(num_groups):
        group_counts[group] = (item_groups[recommendations] == group).float().mean()
    
    # Group fairness metrics
    metrics["group_representation"] = group_counts.cpu().numpy().tolist()
    metrics["max_group_ratio"] = group_counts.max() / group_counts.min()
    metrics["group_entropy"] = -(group_counts * torch.log2(group_counts + 1e-10)).sum().item()
    
    return metrics

def save_recommendations(
    recommendations: torch.Tensor,
    user_ids: List[int],
    item_ids: List[int],
    save_path: Union[str, Path]
) -> None:
    """Save recommendations to file.
    
    Args:
        recommendations (torch.Tensor): Tensor of recommended items
        user_ids (List[int]): List of user IDs
        item_ids (List[int]): List of item IDs
        save_path (Union[str, Path]): Path to save recommendations
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dictionary format
    rec_dict = {}
    for i, user_id in enumerate(user_ids):
        rec_dict[str(user_id)] = [int(item_ids[item]) for item in recommendations[i]]
    
    # Save to JSON
    with open(save_path, "w") as f:
        json.dump(rec_dict, f, indent=4)

def load_recommendations(
    save_path: Union[str, Path]
) -> Tuple[torch.Tensor, List[int]]:
    """Load recommendations from file.
    
    Args:
        save_path (Union[str, Path]): Path to load recommendations from
        
    Returns:
        Tuple[torch.Tensor, List[int]]: Recommendations tensor and user IDs
    """
    save_path = Path(save_path)
    
    # Load from JSON
    with open(save_path, "r") as f:
        rec_dict = json.load(f)
    
    # Convert to tensor
    user_ids = [int(uid) for uid in rec_dict.keys()]
    recommendations = torch.tensor([rec_dict[str(uid)] for uid in user_ids])
    
    return recommendations, user_ids 