#!/usr/bin/env bash
# vecadd_gpu_exact.slurm — matches:
# salloc --partition=gpu_a100 --constraint=rome --mem-per-gpu=100G --ntasks=10 --gres=gpu:1

#SBATCH --job-name=pde_solver
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --mem-per-gpu=100G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --output=slurm-%j.out

set -euo pipefail

module purge >/dev/null 2>&1 || true
module load nvidia/nvhpc-hpcx-cuda12/24.7

# Sanity
hostname
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# Build (expects vector_add.cu in current dir)
nvcc -O2 -arch=sm_80 diff_solver_scratch.cu -o diff_solver_scratch

# Run exactly NTASKS processes (like your salloc)
srun -n "${SLURM_NTASKS}" ./diff_solver_scratch

