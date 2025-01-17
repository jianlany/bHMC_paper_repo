#!/bin/bash

#SBATCH -J HMC_trial_run
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -p htc
#SBATCH -q public
#SBATCH -t 4:00:00

# This file is best used on sol, where python packages are managed by conda/mamba
# One can refer to sol:/home/jianlany/research.yml for environment setup when on other machines
if [ -z $CONDA_DEFAULT_ENV ]; then
    module load mamba/latest
    source activate research
fi

python3 /home/jianlany/development/semicrystalline-generator/master.py 128 -i hmc.ini $1
