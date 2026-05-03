#!/bin/bash
# submit_enriched.sh
# Submits 1 SLURM job: sbert + mlm-only + visit masking + GPT-enriched ICD descriptions.
#
# Usage:
#   bash submit_enriched.sh
#
# Requirements:
#   OPENAI_API_KEY must be set in ~/.bashrc before submitting.
#   The first run calls GPT-4o-mini to enrich ~17K ICD codes (~$2 one-time cost)
#   and saves the result to output_dir/enriched_descriptions.json.
#   All subsequent runs load from that cache with no API calls.

MIMIC_DIR="/gpfs/gibbs/project/wang_zuoheng/yc2256/LLM project"
OUTPUT_DIR="/gpfs/gibbs/project/wang_zuoheng/yc2256/LLM project/outputs_new"
LOG_DIR="/vast/palmer/scratch/wang_zuoheng/yc2256/update_logs"
CODE_DIR="/gpfs/gibbs/project/wang_zuoheng/yc2256/LLM project/Project code"

mkdir -p "$LOG_DIR"

TAG="sbert_mlm_visit_enriched"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=sentemed_${TAG}
#SBATCH --output=${LOG_DIR}/out_${TAG}_%j.txt
#SBATCH --error=${LOG_DIR}/err_${TAG}_%j.txt
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1

source ~/.bashrc
conda activate nlp

export OPENAI_API_KEY="sk-proj-IRkO52_mBaliNWx_D"

cd "${CODE_DIR}"

python main.py \\
  --mimic_dir "${MIMIC_DIR}" \\
  --output_dir "${OUTPUT_DIR}" \\
  --encoder sbert \\
  --masking visit \\
  --mlm_only \\
  --enrich \\
  --openai_model gpt-4o-mini

echo "Done: variant=${TAG}"
EOF

echo "Submitted: $TAG"
echo "Check status with: squeue -u \$USER"
