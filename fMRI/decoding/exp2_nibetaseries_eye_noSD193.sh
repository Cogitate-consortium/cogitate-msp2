#!/bin/bash

# to run type: sbatch exp2_nibetaseries_eye_noSD193.sh


#SBATCH --partition=xnat
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=24000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=zviroth@gmail.com
#SBATCH --time 120:00:00
#SBATCH --job-name=glm
#SBATCH --output=/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/betaseries/slurm-%A_%a.out

module load nibetaseries/0.6.0

singularity run --cleanenv -B /mnt/beegfs:/mnt/beegfs -B /hpc:/hpc ${NIBETASERIES_SIMG} nibs -t VG -c trans_x trans_x_derivative1  trans_x_power2 trans_x_derivative1_power2 trans_y trans_y_derivative1 trans_y_power2 trans_y_derivative1_power2 trans_z trans_z_derivative1 trans_z_power2 trans_z_derivative1_power2 rot_x rot_x_derivative1 rot_x_power2 rot_x_derivative1_power2 rot_y rot_y_derivative1 rot_y_power2 rot_y_derivative1_power2 rot_z rot_z_derivative1 rot_z_power2 rot_z_derivative1_power2 csf white_matter saccades blinks --participant-label SC109 SC123 SD119 SD198 SD135 SD159 SC157 SC189 SC132 SD174 SC154 SD147 SC173 SC196 SD101 SC168 SC124 SC110 SD107 SC145 SC170 SC191 SC194 SC118 SC129 SC114 SC158 SD195 SC122 SC182 SD126 SC172 SD165 SC152 SC121 SD191 SD199 SD190 SC148 SC120 SD136 SC131 SD182 SC187 SC202 SD153 SD131 SD130 SC159 SC140 SC108 SD168 SC143 SD171 SC142 SC136 SD118 SC171 SC183 SD123 SD166 SD134 SD163 SD185 SD188 SD141 SC192 SD137 SC160 SC144 SD194 SD196 --session-label V2 --nthreads 32  --normalize-betas --estimator lss --hrf-model 'spm'  -w /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/betaseries  /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids fmriprep /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/ participant
