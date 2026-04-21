#!/bin/bash
#SBATCH -J parareal-ddp
#SBATCH -A durham 
#SBATCH -p dine2
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Optional: pin to a specific DINE2 node
##SBATCH --nodelist=gc001

# load whatever COSMA modules you actually use here
# module load ...
# source /path/to/venv/bin/activate
source /cosma/apps/do022/dc-dubo2/cuda-env/bin/activate

echo "Node: $(hostname)"
nvidia-smi

torchrun --standalone --nnodes=1 --nproc-per-node=1 distributed_run_parareal.py
