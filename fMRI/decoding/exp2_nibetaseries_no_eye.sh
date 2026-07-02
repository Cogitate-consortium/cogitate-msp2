#!/bin/bash

# to run type: sbatch exp2_nibetaseries_no_eye.sh


#SBATCH --partition=xnat
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=24000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=zviroth@gmail.com
#SBATCH --time 5-12:00:00
#SBATCH --job-name=glm
#SBATCH --output=/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/betaseries/slurm-%A_%a.out

module load nibetaseries/0.6.0

singularity run --cleanenv -B /mnt/beegfs:/mnt/beegfs -B /hpc:/hpc ${NIBETASERIES_SIMG} nibs -t Replay -c trans_x trans_x_derivative1  trans_x_power2 trans_x_derivative1_power2 trans_y trans_y_derivative1 trans_y_power2 trans_y_derivative1_power2 trans_z trans_z_derivative1 trans_z_power2 trans_z_derivative1_power2 rot_x rot_x_derivative1 rot_x_power2 rot_x_derivative1_power2 rot_y rot_y_derivative1 rot_y_power2 rot_y_derivative1_power2 rot_z rot_z_derivative1 rot_z_power2 rot_z_derivative1_power2 csf white_matter --participant-label SD176 SD201 SD156 --session-label V2 --nthreads 32  --normalize-betas --estimator lss --hrf-model 'spm'  -w /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/betaseries  /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids fmriprep /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/ participant
