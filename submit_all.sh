#!/bin/bash
# submit_all.sh
# Submits 6 independent SLURM jobs, one per Sent-e-Med variant.
#
# Usage:
#   bash submit_all.sh
#
# Variants:
#   1. sbert        + nvp+mlm  + code-level masking  (paper default)
#   2. sbert        + mlm-only + code-level masking
#   3. sbert        + mlm-only + visit-level masking
#   4. bio_clinical + nvp+mlm  + code-level masking
#   5. bio_clinical + mlm-only + code-level masking
#   6. bio_clinical + mlm-only + visit-level masking

MIMIC_DIR="/gpfs/gibbs/project/wang_zuoheng/yc2256/LLM project"
OUTPUT_DIR="/gpfs/gibbs/project/wang_zuoheng/yc2256/LLM project/outputs_new"
LOG_DIR="/vast/palmer/scratch/wang_zuoheng/yc2256/update_logs"
CODE_DIR="/gpfs/gibbs/project/wang_zuoheng/yc2256/LLM project/Project code" # adjust if needed

# Make log dir in case it doesn't exist yet
mkdir -p "$LOG_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: submit one variant
#   $1 = short tag used in job name and log filenames (e.g. sbert_nvpmlm_code)
#   $2 = extra python args (e.g. "--encoder sbert --masking code")
# ─────────────────────────────────────────────────────────────────────────────
submit() {
    local TAG="$1"
    local EXTRA_ARGS="$2"

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

cd "${CODE_DIR}"

python main.py \\
  --mimic_dir "${MIMIC_DIR}" \\
  --output_dir "${OUTPUT_DIR}" \\
  ${EXTRA_ARGS}

echo "Done: variant=${TAG}"
EOF

    echo "Submitted: $TAG"
}

# ─────────────────────────────────────────────────────────────────────────────
# Submit all 6 variants
# ─────────────────────────────────────────────────────────────────────────────

# 2. sbert + mlm-only + code masking
submit "sbert_mlm_code" \
  "--encoder sbert --masking code --mlm_only"

# 3. sbert + mlm-only + visit masking
submit "sbert_mlm_visit" \
  "--encoder sbert --masking visit --mlm_only"

# 4. bio_clinical_bert + nvp+mlm + code masking
submit "bio_nvpmlm_code" \
  "--encoder bio_clinical_bert --masking code"

# 5. bio_clinical_bert + mlm-only + code masking
submit "bio_mlm_code" \
  "--encoder bio_clinical_bert --masking code --mlm_only"

# 6. bio_clinical_bert + mlm-only + visit masking
submit "bio_mlm_visit" \
  "--encoder bio_clinical_bert --masking visit --mlm_only"

# 1. sbert + nvp+mlm + code masking  (paper default — no extra flags needed)
submit "sbert_nvpmlm_code" \
  "--encoder sbert --masking code"

echo ""
echo "All 6 jobs submitted. Check status with: squeue -u \$USER"
