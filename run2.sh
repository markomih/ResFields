#!/bin/bash
# template CMD: python launch.py --gpu -1 --config ./configs/dysdf/tnerf.yaml ./configs/dysdf/tnerf.yaml --train dataset.scene=model model.sdf_net.d_hidden=128 model.sdf_net.independent_layers=[1] tag=128ResFields1
set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

export PYTHONUNBUFFERED="True"
EXPERIMENT=$1
CMD=$2
TIME=$(date +"%Y-%m-%d_%H-%M-%S")

OUTPATH=./outputs/$EXPERIMENT/$TIME
VERSION=$(git rev-parse HEAD)

# Save the experiment detail and dir to the common log file
mkdir -p $OUTPATH

LOG="$OUTPATH/$TIME.txt"

# put the arguments on the first line for easy resume
echo -e "$CMD" >> $LOG
echo Logging output to "$LOG"
echo $(pwd) >> $LOG
echo "Version: " $VERSION >> $LOG
echo "Git diff" >> $LOG
echo "" >> $LOG
git diff | tee -a $LOG
echo "" >> $LOG
nvidia-smi | tee -a $LOG

echo -e "\nExecuting: $CMD" >> $LOG
time $CMD 2>&1 | tee -a "$LOG"
