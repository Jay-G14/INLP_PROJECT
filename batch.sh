#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p u22
#SBATCH -n 10
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=inlp_unlearn_%j.log
#SBATCH --mail-type=END
#SBATCH --mail-user=manas.agrawal@research.iiit.ac.in

# --- Environment Setup ---
# PyTorch bundles its own CUDA runtime, so explicit CUDA modules are optional.
# Uncomment below if your PyTorch needs system CUDA (check with: python -c 'import torch; print(torch.cuda.is_available())')
# module load u18/cuda/11.6
# module load u18/cudnn/8.4.0-cuda-11.6

# Activate your virtualenv
source /home2/manas.agrawal/INLP_PROJECT/inlp/bin/activate

# Set HuggingFace cache to /scratch/ (home has 25GB quota, scratch is 2TB)
# But note! compute nodes can't access internet, so model MUST be pre-downloaded
# to this exact HF_HOME first from login node, THEN submit batch script.
export HF_HOME=/scratch/$USER/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Function to send progress emails
send_update() {
    echo "Job ID $SLURM_JOB_ID: $1" | mail -s "INLP Pipeline Update" manas.agrawal@research.iiit.ac.in
}

echo "Starting job on $(hostname) with GPUs: $CUDA_VISIBLE_DEVICES"
send_update "Job started on $(hostname) with GPUs: $CUDA_VISIBLE_DEVICES"

# --- Phase 1: Preprocessing ---
# echo "=== Phase 1: Preprocessing ==="
# python src/data/preprocess.py
# send_update "Phase 1 (Preprocessing) Completed"

# --- Phase 2: SAE Training ---
# Defaults: model=meta-llama/Llama-2-7b-hf, layer=16, expansion=8, k=64, batch=2
echo "=== Phase 2: SAE Training ==="
python src/sae/train.py --epochs 5
send_update "Phase 2 (SAE Training) Completed"

# --- Phase 3: Feature Identification (Diff-in-Means) ---
echo "=== Phase 3: Feature Identification ==="
python src/analysis/diff_means.py --num_features 100 --min_ratio 50.0 --sort_by ratio
send_update "Phase 3 (Feature Identification) Completed"

# --- Phase 4: Unified Evaluation ---
echo "=== Phase 4: Evaluation ==="
python src/eval/unified_evaluate.py --num_features 100 --ablation_scale -3.0 --freq_penalty 1.0 --top_p 0.9
send_update "Phase 4 (Evaluation) Completed"

# --- Phase 5: LLM Judge (optional, requires HF API) ---
echo "=== Phase 5: LLM Judge ==="
python src/eval/evaluate_llm_judge.py --limit 100 || echo "LLM Judge failed (may need HF API access)"
send_update "Phase 5 (LLM Judge) Completed - All Tasks Done"
