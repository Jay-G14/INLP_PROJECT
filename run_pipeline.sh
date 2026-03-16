#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p u22
#SBATCH -n 20 
#SBATCH --exclude=gnode[001-012,016-017,019-023,025-035,037-038,040-041]
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
#SBATCH --output=unlearn.log
#SBATCH --mail-type=END
#SBATCH --mail-user=manas.agrawal@research.iiit.ac.in
#SBATCH --job-name=inlp_pipeline

# Exit on error
set -e

# --- Environment Setup ---
# Set Hugging Face cache to scratch space
# Ada provides 2.0 TB of /scratch per compute node, circumventing permission and quota errors.
export HF_HOME=/scratch/${USER}/hf_cache
export TRANSFORMERS_CACHE=/scratch/${USER}/hf_cache

# PyTorch multi-GPU / accelerate compatibility variables
export OMP_NUM_THREADS=$SLURM_NTASKS
export NCCL_P2P_DISABLE=1 # Sometimes required on HPC clusters with older nodes

# Activate your virtualenvs
source /home2/manas.agrawal/INLP_PROJECT/inlp/bin/activate

# Navigate to the project directory
cd $SLURM_SUBMIT_DIR

# Load HuggingFace token from .env to access gated models (like LLaMA-2)
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
  echo "Loaded environment variables from .env"
else
  echo "WARNING: .env file not found. Hugging Face downloads for gated models may fail."
fi

echo "Starting INLP pipeline at $(date)"
echo "Running on hosts: $SLURM_NODELIST"
echo "Allocated CPUs: $SLURM_NTASKS"
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"

echo "--------------------------------------------------------"
echo "Phase 1: Preprocessing"
echo "--------------------------------------------------------"
# python -u src/data/preprocess.py

echo "--------------------------------------------------------"
echo "Phase 2: SAE Training"
echo "--------------------------------------------------------"
# Added batch size override explicitly along with model params to document intent
python -u src/sae/train.py \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --no_include_target \
    --max_tokens 500000 \
    --layer 12 \
    --expansion_factor 8 \
    --batch_size 4 \
    --epochs 5 \
    --k 32

echo "--------------------------------------------------------"
echo "Phase 3: Feature Identification"
echo "--------------------------------------------------------"
python -u src/analysis/diff_means.py \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --layer 12 \
    --num_features 100 \
    --min_ratio 50.0 \
    --sort_by score

echo "--------------------------------------------------------"
echo "Phase 4: Evaluation"
echo "--------------------------------------------------------"
python -u src/eval/unified_evaluate.py \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --layer 12 \
    --num_features 100 \
    --ablation_scale -3.0 \
    --freq_penalty 1.0 \
    --top_p 0.9

echo "--------------------------------------------------------"
echo "Phase 5: LLM Judge Evaluation"
echo "--------------------------------------------------------"
python -u src/eval/evaluate_llm_judge.py --model Qwen/Qwen2.5-7B-Instruct

echo "--------------------------------------------------------"
echo "Pipeline completed at $(date)"
