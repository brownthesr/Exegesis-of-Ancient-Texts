#!/bin/bash --login 
#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=100G   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH -J "Processing Vectors"   # job name
#SBATCH --qos=cs


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program     doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
source activate script
python3 generation/mean_masks.py -u 
