#!/usr/bin/env bash
#SBATCH --job-name=diff_nsys
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --mem-per-gpu=100G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=nsys-%j.out

set -euo pipefail

module purge >/dev/null 2>&1 || true
module load nvidia/nvhpc-hpcx-cuda12/24.7

hostname
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# Build with profiling-friendly flags
nvcc -O3 -g -lineinfo -arch=sm_80 diff_solver.cu -o diff_solver

# Profile with Nsight Systems
nsys profile \
    --stats=false \
    --force-overwrite=true \
    -o diff_solver_nsys \
    srun -n 1 ./diff_solver

