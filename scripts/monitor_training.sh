#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Training Status Check ===${NC}\n"

# Check training process
if pgrep -f "python src/train.py" > /dev/null; then
    echo -e "${GREEN}✓ Training is running${NC}"
    TRAINING_PID=$(pgrep -f "python src/train.py")
    echo "Process ID: $TRAINING_PID"
else
    echo -e "${RED}✗ Training is not running${NC}"
fi

# Check MLflow server
if pgrep -f "mlflow server" > /dev/null; then
    echo -e "\n${GREEN}✓ MLflow server is running${NC}"
    echo "Access UI at: http://localhost:5000"
else
    echo -e "\n${RED}✗ MLflow server is not running${NC}"
fi

# Show GPU usage
echo -e "\n${YELLOW}=== GPU Usage ===${NC}"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader

# Show latest logs if they exist
if [ -f "training.log" ]; then
    echo -e "\n${YELLOW}=== Latest Logs (last 5 lines) ===${NC}"
    tail -n 5 training.log
else
    echo -e "\n${RED}No training logs found${NC}"
fi

echo -e "\n${YELLOW}To view full logs: tail -f training.log${NC}"
echo -e "${YELLOW}To stop training: kill $TRAINING_PID${NC}" 