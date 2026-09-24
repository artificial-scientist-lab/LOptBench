#!/bin/bash

# Sample Slurm job script for Galvani 

#SBATCH -J celo2_uifo                # Job name
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=a100-galvani   # Which partition will run your job
#SBATCH --time=0-04:30             # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:1               # (optional) Requesting type and number of GPUs
#SBATCH --mem=50G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --output=/mnt/lustre/work/krenn/klz397/celo2_uifo-%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/lustre/work/krenn/klz397/celo2_uifo-%j.err        # File to which STDERR will be written - make sure this is not on $HOME
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
source "$(conda info --base)/etc/profile.d/conda.sh"
#conda activate /mnt/lustre/work/krenn/klz397/.conda/py-312-veloenv
PY=/mnt/lustre/work/krenn/klz397/.conda/py-312-veloenv/bin/python

cd /home/krenn/klz397/LOptBench
$PY -c "import jax; assert jax.default_backend() == 'gpu', jax.devices(); print(jax.devices())"
$PY scripts/uifo_celo2_2.py
