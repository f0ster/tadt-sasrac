#!/bin/bash

# Exit on error
set -e

echo "Starting training..."

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create necessary directories
mkdir -p checkpoints
mkdir -p logs

# Run training directly
python src/train.py 2>&1 | tee logs/training.log

echo "Training completed. Check logs/training.log for details." 