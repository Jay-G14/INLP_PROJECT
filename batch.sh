#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p u22
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=inlp_unlearn_%j.log
#SBATCH --mail-type=END
#SBATCH --mail-user=manas.agrawal@research.iiit.ac.in

# 1. Setup environment
module load u22/python/3.12.4
module load u22/cuda/12.1

# Use the exact path from your pwd command
source /home2/manas.agrawal/INLP_PROJECT/inlp_env/bin/activate

# Function to send progress emails manually
send_update() {
    echo "Job ID $SLURM_JOB_ID: $1" | mail -s "INLP Pipeline Update" manas.agrawal@research.iiit.ac.in
}

send_update "Starting the job:"

# Phase 1: Preprocessing
echo "Starting Phase 1..."
python src/data/preprocess.py
send_update "Phase 1 (Preprocessing) Completed"

# Phase 2: SAE Training
echo "Starting Phase 2..."
python src/sae/train.py --model gemma-2b --layer 12 --use_neutral_corpus --epochs 5
send_update "Phase 2 (SAE Training) Completed"

# Phase 3: Feature Identification
echo "Starting Phase 4..."
python src/analysis/diff_means.py --model gemma-2b --layer 12 --method sparsity
send_update "Phase 3 (Feature Identification) Completed"

# Phase 4: Evaluation
echo "Starting Phase 4..."
python src/eval/evaluate.py --model gemma-2b --layer 12 --limit 50
python src/eval/evaluate_mmlu.py --model gemma-2b --layer 12 --limit 50
python src/eval/evaluate_perplexity.py --model gemma-2b --layer 12 --limit 50
python src/eval/evaluate_comprehensive.py --model gemma-2b --layer 12 --limit 50
send_update "Phase 4 (Evaluation) Completed - All Tasks Done"
