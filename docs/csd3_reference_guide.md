# ─────────────────────────────────────────────────────────────────────────────
# CSD3 HPC Reference Guide — Common Commands
# University of Cambridge
# Built from real experience — tm922, May 2026
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ENVIRONMENT SETUP
# ═══════════════════════════════════════════════════════════════════════════════

# Activate conda environment (use eval method — NOT source)
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate OAIKaggle              # Python 3.11 — works on GPU nodes
conda activate mrkr_env               # Python 3.10 — CPU jobs only

# List all environments
conda env list

# Create new environment with specific Python version
conda create -n new_env python=3.11 -y

# Clone existing environment
conda create --name NewName --clone OldName

# Remove environment
conda env remove --name OldName

# Install packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scikit-learn pillow tqdm seaborn matplotlib pydicom pylibjpeg pylibjpeg-libjpeg

# Check installed packages
pip list
pip list | grep -E "torch|pandas|numpy|sklearn"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SLURM JOB MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

# Submit a job
sbatch ~/mrkr_klg/code/slurm_train.sh resnet50

# Check your jobs
squeue -u tm922

# Check all jobs on a partition
squeue -p ampere

# Check estimated start time
squeue -u tm922 --start

# Cancel a job
scancel JOBID

# Cancel multiple jobs
scancel 12345 12346 12347

# Cancel all your jobs
scancel -u tm922

# Check job details (time, node, status)
scontrol show job JOBID

# Check time remaining on a running job
scontrol show job JOBID | grep -i time

# Extend a job time limit (only works before job starts)
scontrol update JobId=JOBID TimeLimit=06:00:00

# Check your allocation balance
mybalance

# Check your account/partition access
sacctmgr show user tm922 withassoc format=user,account,partition

# Watch queue refreshing every 30 seconds
watch -n 30 squeue -u tm922


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PARTITION AND NODE STATUS
# ═══════════════════════════════════════════════════════════════════════════════

# Check all partitions
sinfo

# Check GPU partitions only
sinfo | grep -E "idle|mix" | grep -v cpu

# Check ampere partition specifically
sinfo -p ampere

# Check your position in queue
squeue -p ampere --sort=+i | grep -n tm922

# Count jobs waiting on ampere
squeue -p ampere | wc -l


# ═══════════════════════════════════════════════════════════════════════════════
# 4. STORAGE AND FILE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

# Check storage quota
quota -s /rds/user/tm922

# Check disk usage
df -h /rds/user/tm922/hpc-work
du -sh /rds/user/tm922/hpc-work/data/
du -sh /rds/user/tm922/hpc-work/data/*/

# Find files
find /rds/user/tm922 -name "MRKR_image_metadata.csv" 2>/dev/null
find ~/mrkr_klg -name "*.sh"
find ~/mrkr_klg -name "*.dcm" | wc -l
find ~/mrkr_klg -name "*.png" | wc -l

# Count files
find /path/to/folder -name "*.png" | wc -l

# Transfer files from Mac to HPC (run on Mac)
scp /local/path/file.py tm922@login-p-4.hpc.cam.ac.uk:~/mrkr_klg/code/

# Transfer files from HPC to Mac (run on Mac)
scp tm922@login-p-4.hpc.cam.ac.uk:~/mrkr_klg/eda/*.png ./figures/

# Check file content
cat filename.json
head -5 filename.csv
tail -20 filename.log
tail -f filename.log          # follow live log


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CONDA MODULES
# ═══════════════════════════════════════════════════════════════════════════════

# Load CUDA (required for GPU jobs)
module load cuda/11.8           # matches PyTorch cu118 — use this one
module load cuda/12.1           # alternative — only if using cu121

# Search available modules
module avail 2>&1 | grep -i cuda
module avail 2>&1 | grep -i python
module avail 2>&1 | grep -i pytorch
module avail 2>&1 | grep -i aws

# Show module details
module show ceuadmin/awscli/2.27.52


# ═══════════════════════════════════════════════════════════════════════════════
# 6. AWS S3 (MRKR DATASET)
# ═══════════════════════════════════════════════════════════════════════════════

# Install AWS CLI in conda environment
pip install awscli

# Check AWS CLI works
aws --version

# Test S3 access
aws s3 ls s3://emory-mrkr-dataset/ --region us-east-1

# List tables
aws s3 ls s3://emory-mrkr-dataset/tables/ --region us-east-1

# Download a single file
aws s3 cp s3://emory-mrkr-dataset/tables/MRKR_demographics.csv \
    /rds/user/tm922/hpc-work/data/mrkr/MRKR_demographics.csv \
    --region us-east-1


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DEBUGGING
# ═══════════════════════════════════════════════════════════════════════════════

# Check Python syntax before submitting
python3 -m py_compile script.py && echo "OK"

# Test script on tiny subset before GPU job
python3 -c "
import pandas as pd
df = pd.read_csv('/path/to/manifest.csv')
sample = df.groupby('label').head(10)
sample.to_csv('/home/tm922/mrkr_klg/data/smoke_50.csv', index=False)
print('Saved', len(sample), 'rows')
"

# Check job error log
cat ~/mrkr_klg/logs/train_JOBID_mrkr_train.err

# Check job output log
cat ~/mrkr_klg/logs/train_JOBID_mrkr_train.out
tail -20 ~/mrkr_klg/logs/train_JOBID_mrkr_train.out

# Check GPU availability in Python
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"

# Check CPU architecture
uname -m
cat /proc/cpuinfo | grep "model name" | head -1


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PROJECT PATHS — MRKR PROJECT
# ═══════════════════════════════════════════════════════════════════════════════

# Project root
# ~/mrkr_klg/

# Code
# ~/mrkr_klg/code/

# EDA outputs
# ~/mrkr_klg/eda/

# Training results
# ~/mrkr_klg/runs/{resnet50,densenet121,efficientnet_b0}/

# Logs
# ~/mrkr_klg/logs/

# Data (on RDS)
# /rds/user/tm922/hpc-work/data/mrkr/           — MRKR CSVs and metadata
# /rds/user/tm922/hpc-work/data/mrkr_png_v2/    — 10K PNGs (training dataset)
# /rds/user/tm922/hpc-work/data/mrkr_png_v3/    — 20K PNGs (future use)
# /rds/user/tm922/hpc-work/data/knee_oa/         — OAI/Kaggle dataset

# Key CSV files
# /rds/user/tm922/hpc-work/data/mrkr/mrkr_selected_v2.csv   — 10K working set
# /rds/user/tm922/hpc-work/data/mrkr_png_v2/mrkr_png_manifest.csv — training manifest
# /rds/user/tm922/hpc-work/data/mrkr_png_v3/mrkr_png_manifest_v3.csv — v3 manifest


# ═══════════════════════════════════════════════════════════════════════════════
# 9. KNOWN ISSUES AND FIXES
# ═══════════════════════════════════════════════════════════════════════════════

# ISSUE: "Illegal instruction" on GPU nodes
# CAUSE: Python 3.10 conda environment incompatible with ampere node CPU
# FIX:   Use OAIKaggle environment (Python 3.11)

# ISSUE: "Invalid account or account/partition combination"
# CAUSE: Wrong partition/account combination
# FIX:   GPU jobs → partition=ampere, account=TORABI-SL3-GPU
#        CPU jobs → partition=icelake, account=TORABI-SL3-CPU

# ISSUE: Job stuck in PD with "ReqNodeNotAvail, Reserved for maintenance"
# CAUSE: Partition under maintenance
# FIX:   Switch to another partition (icelake instead of cclake)

# ISSUE: Job killed before finishing
# CAUSE: Time limit exceeded
# FIX:   Cannot extend running job — set generous time upfront
#        Script is idempotent — resubmit and it will skip completed work

# ISSUE: "KeyError: 'KLG'" in conversion script
# CAUSE: Column renamed from KLG to label in filter script
# FIX:   Always test script locally on 5 rows before submitting SLURM job

# ISSUE: DICOM "Unable to decompress JPEG Lossless" error
# CAUSE: Missing pylibjpeg codec
# FIX:   pip install pylibjpeg pylibjpeg-libjpeg

# ISSUE: AWS CLI "Permission denied"
# CAUSE: System module not executable for your user
# FIX:   pip install awscli (installs in your conda environment)

# ISSUE: "images/images/" doubled path in PNG loading
# CAUSE: img_root already contains "images/" and png_path also starts with "images/"
# FIX:   Set --img_root to parent folder not images subfolder


# ═══════════════════════════════════════════════════════════════════════════════
# 10. WORKFLOW CHECKLIST FOR NEW ML PROJECT
# ═══════════════════════════════════════════════════════════════════════════════

# 1. Create project folder structure
#    mkdir -p ~/project/{code,data,eda,runs,logs,results,notebooks}

# 2. Activate OAIKaggle environment
#    conda activate OAIKaggle

# 3. Write and test script locally on 5-10 rows
#    python3 -m py_compile script.py && echo "OK"
#    python3 script.py --small_test_args

# 4. Submit GPU smoke test (15 min job) before full training
#    sbatch slurm_gpu_test.sh
#    cat logs/gpu_test_JOBID.out   # verify CUDA: True

# 5. Only then submit full training jobs
#    sbatch slurm_train.sh model_name

# 6. Monitor
#    squeue -u tm922
#    tail -f logs/train_JOBID.out

# 7. Check results
#    cat runs/model/test_metrics.json
