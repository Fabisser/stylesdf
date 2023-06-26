#!/bin/bash


#SBATCH --job-name="animatennfm"
#SBATCH --time=1440
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --account=education-abe-msc-g

module load 2022r2 python py-pip


srun python3 run_IDR.py --conf DTU_style.conf --scan_id 24

srun python3 run_style.py --conf DTU_style.conf --scan_id 24
