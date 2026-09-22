#!/bin/bash

# Sample Slurm job script for Galvani 

#SBATCH -J velo_uifo                # Job name
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=2080-galvani   # Which partition will run your job
#SBATCH --time=0-01:05             # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:2               # (optional) Requesting type and number of GPUs
#SBATCH --mem=50G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --output=/mnt/lustre/work/krenn/klz397/velo_uifo-%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/lustre/work/krenn/klz397/velo_uifo-%j.err        # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=yuliya.barko@student.uni-tuebingen.de   # Email to which notifications will be sent

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus
ls $WORK # not necessary just here to illustrate that $WORK is available here

# Setup Phase
# add possibly other setup code here, e.g.
# - copy singularity images or datasets to local on-compute-node storage like /scratch_local
# - loads virtual envs, like with anaconda
# - set environment variables
# - determine commandline arguments for `srun` calls

# Compute Phase
# srun python3 runfile.py  # srun will automatically pickup the configuration defined via `#SBATCH` and `sbatch` command line arguments
conda activate /mnt/lustre/work/krenn/klz397/.conda/py-312-veloenv
cd /home/krenn/klz397/LOptBench
python -c "import jax; print(jax.devices())"   # should show a CudaDevice, not only CPU
python scripts/uifo_velo_vs_adam.py