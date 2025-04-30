import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List
import argparse
import json
from pathlib import Path
import torch.optim as optim
import math
import torch.cuda.amp as amp
import mlflow
from datetime import datetime

from data.dataset import get_loader
from models.tadt_rec import TADTRec
from utils.optimization import optimize_for_nvidia, get_nvidia_optimizer
from utils.mlflow_utils import MLflowLogger

class TADTRecTrainer(pl.LightningModule):
    def __init__(self, config: Dict[str, Any], mlflow_logger: Optional[MLflowLogger] = None):
        super().__init__()
        self.config = config
        self.mlflow_logger = mlflow_logger
        
        # Initialize model with only the parameters it accepts
        self.model = TADTRec(
            n_items=config["n_items"],
            d=config.get("d", 128),
            K=config.get("K", 6),
            max_len=config.get("max_len", 40)
        ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Apply NVIDIA optimizations
        optimize_for_nvidia(self.model)
        
        # Save hyperparameters
        self.save_hyperparameters(config)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        self.automatic_optimization = False
        
        # Warmup steps
        self.warmup_steps = config.get("warmup_steps", 1000)
        
        # Log parameters to MLflow if logger is provided
        if mlflow_logger:
            mlflow_logger.log_params(config)
    
    def forward(self, x, lengths):
        return self.model(x, lengths)
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        
        # Zero gradients
        opt.zero_grad()
        
        # Get batch data
        center, ctx, seqlen = batch
        batch_size = center.size(0)
        
        # Forward pass with gradient scaling for mixed precision
        with torch.cuda.amp.autocast():
            # Get logits and ensure they match the number of classes
            logits, _ = self(center.unsqueeze(1), torch.ones_like(seqlen))  # [batch_size, n_items]
            
            # Debug info
            if batch_idx == 0:
                print(f"Logits shape: {logits.shape}")
                print(f"Target (ctx) shape: {ctx.shape}")
                print(f"Target (ctx) min: {ctx.min()}, max: {ctx.max()}")
                print(f"Number of classes (n_items): {self.config['n_items']}")
            
            # Ensure targets are valid
            ctx = torch.clamp(ctx, 0, self.config['n_items'] - 1)
            
            # Compute loss
            ce_loss = self.criterion(logits, ctx)
        
        # Backward pass with gradient scaling
        self.manual_backward(ce_loss)
        opt.step()
        
        # Log metrics
        metrics = {
            'loss': ce_loss.item()
        }
        
        # Log to MLflow if logger is provided
        if self.mlflow_logger:
            self.mlflow_logger.log_metrics(metrics, step=self.global_step)
        
        return ce_loss
    
    def configure_optimizers(self):
        # Use NVIDIA-optimized optimizer
        optimizer = get_nvidia_optimizer(
            self.model.parameters(),
            lr=self.config.get("lr", 1e-3),
            weight_decay=self.config.get("weight_decay", 0.01)
        )
        
        # Learning rate scheduler with warmup
        def lr_lambda(step):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return 0.1 + 0.9 * math.exp(-0.1 * (step - self.warmup_steps))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

def main() -> None:
    """Main training function."""
    # Load config
    with open("configs/tadt_rec_config.json", "r") as f:
        config = json.load(f)
    
    # Set CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision('high')  # Optimize for Tensor Cores
    
    # Initialize MLflow logger
    mlflow_logger = MLflowLogger(
        experiment_name="tadt_rec_experiments",
        tracking_uri="file:./mlruns",  # Local tracking
        artifact_location="./mlruns"
    )
    
    # Prepare data
    train_dl, n_items = get_loader(
        "data/processed/fb_walks.npz",
        batch_size=config["batch_size"]
    )
    config["n_items"] = n_items
    
    # Initialize model and trainer
    model = TADTRecTrainer(config, mlflow_logger)
    
    # Callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="loss",
            mode="min",
            save_top_k=3,
            dirpath="checkpoints",
            filename="tadt-{epoch:02d}-{loss:.4f}"
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step")
    ]
    
    # Logger
    logger = pl.loggers.TensorBoardLogger("lightning_logs", name="tadt_moe")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator="gpu",
        devices=1,
        precision="16-mixed",  # Use mixed precision
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
    )
    
    # Train the model
    trainer.fit(model, train_dl)
    
    # Log final model and artifacts
    if mlflow_logger:
        # Log the final model
        mlflow_logger.log_model(model.model, "final_model")
        
        # Log checkpoints
        for checkpoint in Path("checkpoints").glob("*.ckpt"):
            mlflow_logger.log_checkpoint(str(checkpoint))
        
        # Log config and other artifacts
        mlflow_logger.log_artifacts("configs")
        mlflow_logger.log_artifacts("lightning_logs")
        
        # End the run
        mlflow_logger.end_run()

if __name__ == "__main__":
    main()