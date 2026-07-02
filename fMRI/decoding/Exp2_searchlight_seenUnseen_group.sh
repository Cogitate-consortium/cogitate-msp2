#!/bin/bash
# sbatch Exp2_searchlight_seenUnseen_group.sh
#SBATCH --partition=xnat
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=zviroth@gmail.com
#SBATCH --time 12:00:00
#SBATCH --output=./slurm/slurm-seenUnseen-group-%A_%a.out
#SBATCH --job-name=snUnsGrp


cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Exp2/Zvi/slurm

module purge; module load Anaconda3/2020.11; source /hpc/shared/EasyBuild/apps/Anaconda3/2020.11/bin/activate; conda activate /hpc/users/$USER/.conda/envs/mne_ecog01

python3 ../Exp2_searchlight_seenUnseen_group_forSlurm.py