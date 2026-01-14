#!/bin/bash
#SBATCH --job-name=python_job
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH -p cosma7
#SBATCH -A do009
#SBATCH --mem=4G

# Optional: load modules if your cluster requires it
#module load python/3.13

# Activate virtual environment
source ../../vesicles/Ves2Dpy/venv/bin/activate 
source ../../vesicles/Ves2Dpy/venv/bin/activate

# Run Python script
python newtorch/run_parareal.py

