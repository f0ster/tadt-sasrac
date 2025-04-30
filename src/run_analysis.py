import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.tadt_rec import TADTRec
from src.utils.report import ReportGenerator
from src.utils.analysis import (
    analyze_model_parameters,
    analyze_model_complexity,
    analyze_training_speed,
    analyze_memory_usage,
    analyze_expert_usage
)
from src.utils.visualization import (
    plot_training_metrics,
    visualize_model_architecture,
    plot_attention_patterns,
    plot_expert_usage,
    plot_recommendation_distribution
)
from src.utils.recommendation import (
    generate_recommendations,
    evaluate_recommendations,
    analyze_recommendation_diversity,
    analyze_recommendation_fairness
)

def main():
    print("Starting TADT-SASRec model analysis...")
    
    # Load configuration
    with open("configs/tadt_rec_config.json", "r") as f:
        config = json.load(f)
    
    # Initialize model
    print("Loading model...")
    model = TADTRec(
        n_items=config["n_items"],
        d=config["d"],
        K=config["K"],
        max_len=config["max_len"],
        dropout=config["dropout"],
        num_experts=config["num_experts"],
        top_k=config["top_k"]
    )
    
    # Load trained weights
    checkpoint = torch.load("lightning_logs/version_3/checkpoints/epoch=4-step=145945.ckpt")
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load data
    print("Loading data...")
    data = np.load("data/processed/fb_walks.npz")
    train_data = torch.tensor(data["train"])
    val_data = torch.tensor(data["val"])
    test_data = torch.tensor(data["test"])
    
    # Initialize report generator
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    report_generator = ReportGenerator(report_dir, model, "tadt_rec")
    
    # Generate model analysis report
    print("Generating model analysis report...")
    input_shape = (1, config["max_len"])
    model_report = report_generator.generate_model_report(input_shape)
    
    # Print model metrics
    print("\nModel Analysis Results:")
    print(f"Total Parameters: {model_report['parameter_analysis']['total_params']:,}")
    print(f"Trainable Parameters: {model_report['parameter_analysis']['trainable_params']:,}")
    print(f"FLOPs: {model_report['complexity_analysis']['flops']:,}")
    print(f"Peak Memory Usage: {model_report['memory_analysis']['peak_memory'] / 1024**2:.2f} MB")
    
    # Generate training analysis report
    print("\nGenerating training analysis report...")
    metrics_df = pd.read_csv("lightning_logs/version_3/metrics.csv")
    train_metrics = {"loss": metrics_df["loss"].tolist()}
    training_report = report_generator.generate_training_report(train_metrics)
    
    # Plot training metrics
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df["loss"].dropna(), label="Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(report_dir / "training_loss.png")
    plt.close()
    
    # Generate recommendation analysis report
    print("\nGenerating recommendation analysis report...")
    recommendations = generate_recommendations(
        model,
        test_data,
        k=20,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    recommendation_report = report_generator.generate_recommendation_report(
        test_data,
        test_data[:, -1],
        k_values=[5, 10, 20]
    )
    
    # Print recommendation metrics
    print("\nRecommendation Analysis Results:")
    for k in [5, 10, 20]:
        print(f"\nMetrics @ {k}:")
        print(f"Hit Rate: {recommendation_report['evaluation_metrics'][f'hit_rate@{k}']:.4f}")
        print(f"NDCG: {recommendation_report['evaluation_metrics'][f'ndcg@{k}']:.4f}")
        print(f"Recall: {recommendation_report['evaluation_metrics'][f'recall@{k}']:.4f}")
    
    # Generate summary report
    print("\nGenerating summary report...")
    summary_path = report_generator.generate_summary_report()
    print(f"Summary report generated at: {summary_path}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 