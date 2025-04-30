import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns

from .analysis import (
    analyze_model_parameters,
    analyze_model_complexity,
    analyze_training_speed,
    analyze_memory_usage,
    analyze_expert_usage,
    analyze_recommendation_quality
)
from .visualization import (
    plot_training_metrics,
    visualize_model_architecture,
    plot_attention_patterns,
    plot_expert_usage,
    plot_recommendation_distribution,
    plot_learning_curves,
    plot_memory_usage
)
from .recommendation import (
    generate_recommendations,
    evaluate_recommendations,
    analyze_recommendation_diversity,
    analyze_recommendation_fairness
)

class ReportGenerator:
    """Generates comprehensive reports for model analysis and evaluation."""
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        model: nn.Module,
        model_name: str = "model"
    ):
        """Initialize report generator.
        
        Args:
            output_dir (Union[str, Path]): Directory to save reports
            model (nn.Module): Model to analyze
            model_name (str): Name of the model for report identification
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.model_name = model_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create subdirectories
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
    
    def generate_model_report(
        self,
        input_shape: tuple,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """Generate model analysis report.
        
        Args:
            input_shape (tuple): Shape of input tensor
            device (str): Device to run analysis on
            
        Returns:
            Dict[str, Any]: Model analysis results
        """
        results = {}
        
        # Analyze parameters
        param_stats = analyze_model_parameters(self.model)
        results["parameter_analysis"] = param_stats
        
        # Analyze complexity
        complexity = analyze_model_complexity(self.model, input_shape, device)
        results["complexity_analysis"] = complexity
        
        # Analyze memory usage
        memory = analyze_memory_usage(self.model, input_shape, device=device)
        results["memory_analysis"] = memory
        
        # Visualize architecture
        arch_path = self.plots_dir / f"{self.model_name}_architecture.png"
        visualize_model_architecture(self.model, input_shape, str(arch_path))
        
        # Save results
        self._save_results(results, "model_analysis")
        
        return results
    
    def generate_training_report(
        self,
        train_metrics: Dict[str, List[float]],
        val_metrics: Optional[Dict[str, List[float]]] = None,
        train_losses: Optional[List[float]] = None,
        val_losses: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Generate training analysis report.
        
        Args:
            train_metrics (Dict[str, List[float]]): Training metrics
            val_metrics (Optional[Dict[str, List[float]]]): Validation metrics
            train_losses (Optional[List[float]]): Training losses
            val_losses (Optional[List[float]]): Validation losses
            
        Returns:
            Dict[str, Any]: Training analysis results
        """
        results = {}
        
        # Plot metrics
        metrics_path = self.plots_dir / f"{self.model_name}_training_metrics.png"
        plot_training_metrics(train_metrics, str(metrics_path))
        
        # Plot learning curves if losses are provided
        if train_losses is not None and val_losses is not None:
            curves_path = self.plots_dir / f"{self.model_name}_learning_curves.png"
            plot_learning_curves(
                train_losses,
                val_losses,
                train_metrics,
                val_metrics,
                str(curves_path)
            )
        
        # Save results
        self._save_results(results, "training_analysis")
        
        return results
    
    def generate_recommendation_report(
        self,
        user_sequences: torch.Tensor,
        ground_truth: torch.Tensor,
        k_values: List[int] = [5, 10, 20],
        item_popularity: Optional[torch.Tensor] = None,
        item_groups: Optional[torch.Tensor] = None,
        num_groups: Optional[int] = None,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """Generate recommendation analysis report.
        
        Args:
            user_sequences (torch.Tensor): User sequences
            ground_truth (torch.Tensor): Ground truth items
            k_values (List[int]): List of k values for metrics
            item_popularity (Optional[torch.Tensor]): Item popularity scores
            item_groups (Optional[torch.Tensor]): Item group memberships
            num_groups (Optional[int]): Number of groups
            device (str): Device to run analysis on
            
        Returns:
            Dict[str, Any]: Recommendation analysis results
        """
        results = {}
        
        # Generate recommendations
        recommendations = generate_recommendations(
            self.model,
            user_sequences,
            max(k_values),
            device=device
        )
        
        # Evaluate recommendations
        eval_metrics = evaluate_recommendations(
            recommendations,
            ground_truth,
            k_values
        )
        results["evaluation_metrics"] = eval_metrics
        
        # Analyze diversity
        if item_popularity is not None:
            diversity = analyze_recommendation_diversity(
                recommendations,
                item_popularity
            )
            results["diversity_analysis"] = diversity
            
            # Plot distribution
            dist_path = self.plots_dir / f"{self.model_name}_recommendation_distribution.png"
            plot_recommendation_distribution(
                recommendations,
                item_popularity,
                save_path=str(dist_path)
            )
        
        # Analyze fairness
        if item_groups is not None and num_groups is not None:
            fairness = analyze_recommendation_fairness(
                recommendations,
                item_groups,
                num_groups
            )
            results["fairness_analysis"] = fairness
        
        # Save results
        self._save_results(results, "recommendation_analysis")
        
        return results
    
    def _save_results(
        self,
        results: Dict[str, Any],
        report_type: str
    ) -> None:
        """Save analysis results to file.
        
        Args:
            results (Dict[str, Any]): Results to save
            report_type (str): Type of report
        """
        # Convert tensors to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                serializable_results[key] = value.cpu().numpy().tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {
                    k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        # Save to JSON
        save_path = self.data_dir / f"{self.model_name}_{report_type}_{self.timestamp}.json"
        with open(save_path, "w") as f:
            json.dump(serializable_results, f, indent=4)
    
    def generate_summary_report(self) -> str:
        """Generate a summary markdown report.
        
        Returns:
            str: Path to the summary report
        """
        summary_path = self.output_dir / f"{self.model_name}_summary_{self.timestamp}.md"
        
        with open(summary_path, "w") as f:
            f.write(f"# {self.model_name} Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Add links to all generated files
            f.write("## Generated Files\n\n")
            for file in self.output_dir.glob("**/*"):
                if file.is_file():
                    rel_path = file.relative_to(self.output_dir)
                    f.write(f"- [{rel_path}]({rel_path})\n")
        
        return str(summary_path) 