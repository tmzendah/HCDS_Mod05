#!/bin/bash
#SBATCH --job-name=mrkr_v3
#SBATCH --output=/home/tm922/mrkr_klg/logs/v3_%j_%x.out
#SBATCH --error=/home/tm922/mrkr_klg/logs/v3_%j_%x.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=ampere
#SBATCH --account=TORABI-SL3-GPU

# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   sbatch slurm_train_v3.sh resnet50 42
#   sbatch slurm_train_v3.sh resnet50 123
#   sbatch slurm_train_v3.sh resnet50 456
#   sbatch slurm_train_v3.sh efficientnet_b0 42
#   sbatch slurm_train_v3.sh efficientnet_b0 123
#   sbatch slurm_train_v3.sh efficientnet_b0 456
#
# Submit all 6 at once:
#   for model in resnet50 efficientnet_b0; do
#       for seed in 42 123 456; do
#           sbatch slurm_train_v3.sh $model $seed
#       done
#   done
# ─────────────────────────────────────────────────────────────────────────────

MODEL=${1:-resnet50}
SEED=${2:-42}

echo "============================================================"
echo "  MRKR v3 Training"
echo "  Model   : $MODEL"
echo "  Seed    : $SEED"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $SLURMD_NODENAME"
echo "  Started : $(date)"
echo "============================================================"

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR=/home/tm922/mrkr_klg
DATA_CSV=/rds/user/tm922/hpc-work/data/mrkr_cropped/mrkr_cropped_manifest.csv
IMG_ROOT=/rds/user/tm922/hpc-work/data/mrkr_cropped
OUTPUT_DIR=$PROJECT_DIR/runs/v3_${MODEL}_seed${SEED}
LOG_DIR=$PROJECT_DIR/logs

mkdir -p $OUTPUT_DIR $LOG_DIR

# ── Environment ───────────────────────────────────────────────────────────────
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate OAIKaggle
module load cuda/11.8

# ── GPU check ─────────────────────────────────────────────────────────────────
echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "
import torch
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU     : {torch.cuda.get_device_name(0)}')
"
echo ""

# ── Verify manifest exists ────────────────────────────────────────────────────
if [ ! -f "$DATA_CSV" ]; then
    echo "ERROR: Manifest not found: $DATA_CSV"
    exit 1
fi

echo "Manifest: $DATA_CSV"
echo "Output  : $OUTPUT_DIR"
echo ""

# ── Training ──────────────────────────────────────────────────────────────────
echo "Starting training: $MODEL seed=$SEED"
echo ""

python3 $PROJECT_DIR/code/mrkr_train_v3.py \
    --data_csv      $DATA_CSV \
    --img_root      $IMG_ROOT \
    --output_dir    $OUTPUT_DIR \
    --model         $MODEL \
    --num_classes   5 \
    --batch_size    32 \
    --epochs        30 \
    --lr            1e-4 \
    --weight_decay  1e-4 \
    --num_workers   8 \
    --seed          $SEED \
    --val_size      0.15 \
    --test_size     0.15 \
    --patience      7

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  Training complete"
echo "  Model     : $MODEL"
echo "  Seed      : $SEED"
echo "  Exit code : $EXIT_CODE"
echo "  Finished  : $(date)"
echo "  Results   : $OUTPUT_DIR/test_metrics.json"
echo "============================================================"

exit $EXIT_CODE
