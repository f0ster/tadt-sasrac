import os
import mlflow
import torch
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

class MLflowLogger:
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        """Initialize MLflow logger.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI (default: local)
            artifact_location: Location to store artifacts (default: mlruns)
        """
        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
                experiment = mlflow.get_experiment(experiment_id)
        except Exception as e:
            print(f"Error creating/getting experiment: {e}")
            print("Creating new experiment...")
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
            experiment = mlflow.get_experiment(experiment_id)
        
        self.experiment_id = experiment.experiment_id
        
        # Create run
        self.run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model: torch.nn.Module, model_name: str) -> None:
        """Log PyTorch model to MLflow."""
        mlflow.pytorch.log_model(model, model_name)
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log artifacts to MLflow."""
        mlflow.log_artifacts(local_dir, artifact_path)
    
    def log_checkpoint(self, checkpoint_path: str, artifact_path: Optional[str] = None) -> None:
        """Log checkpoint to MLflow."""
        mlflow.log_artifact(checkpoint_path, artifact_path)
    
    def end_run(self) -> None:
        """End the current run."""
        mlflow.end_run()
    
    @staticmethod
    def get_best_run(experiment_name: str, metric: str, mode: str = "min") -> Dict:
        """Get the best run from an experiment based on a metric.
        
        Args:
            experiment_name: Name of the experiment
            metric: Metric to optimize
            mode: "min" or "max" to determine best value
            
        Returns:
            Dictionary containing run information
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'DESC' if mode == 'max' else 'ASC'}"]
        )
        
        if runs.empty:
            raise ValueError(f"No runs found for experiment {experiment_name}")
        
        return runs.iloc[0].to_dict()
    
    @staticmethod
    def load_model(run_id: str, model_name: str) -> torch.nn.Module:
        """Load a model from a specific run.
        
        Args:
            run_id: ID of the run containing the model
            model_name: Name of the model artifact
            
        Returns:
            Loaded PyTorch model
        """
        return mlflow.pytorch.load_model(f"runs:/{run_id}/{model_name}") 