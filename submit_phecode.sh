#!/bin/bash
# submit_phecode.sh
# Submits 2 independent SLURM jobs for PheCode dual-embedding variants,
# both built on top of sbert_mlm_visit (the best-performing baseline).
#
# Usage:
#   bash submit_phecode.sh
#
# Variants:
#   1. sbert + mlm-only + visit masking + PheCode concat fusion
#   2. sbert + mlm-only + visit masking + PheCode gate fusion

MIMIC_DIR="/gpfs/gibbs/project/wan/yc22/LLM project"
OUTPUT_DIR="/gpfs/gibbs/project/wan/yc22/LLM project/outputs_new"
LOG_DIR="/vast/palmer/scratch/wan/yc22/update_logs"
CODE_DIR="/gpfs/gibbs/project/wan/yc22/LLM project/Project code"
PHECODE_MAP="/gpfs/gibbs/project/wan/yc22/LLM project/Project code/phecodes_cm_rolled.csv"

mkdir -p "$LOG_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: submit one variant
#   $1 = short tag used in job name and log filenames
#   $2 = extra python args
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
  --encoder sbert \\
  --masking visit \\
  --mlm_only \\
  --phecode \\
  --phecode_map "${PHECODE_MAP}" \\
  ${EXTRA_ARGS}

echo "Done: variant=${TAG}"
EOF

    echo "Submitted: $TAG"
}

# ─────────────────────────────────────────────────────────────────────────────
# Submit both PheCode variants
# ─────────────────────────────────────────────────────────────────────────────

# 1. sbert + mlm-only + visit masking + PheCode concat fusion
submit "sbert_mlm_visit_phecodeconcат" \
  "--phecode_fusion concat"

# 2. sbert + mlm-only + visit masking + PheCode gate fusion
submit "sbert_mlm_visit_phecodegate" \
  "--phecode_fusion gate"

echo ""
echo "Both PheCode jobs submitted. Check status with: squeue -u \$USER"
