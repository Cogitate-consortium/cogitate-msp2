#!/bin/bash
#SBATCH --partition=xnat
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=o.ferrante@bham.ac.uk
#SBATCH --time 12:00:00
#SBATCH --chdir=/hpc/users/oscar.ferrante/git/MEG_cogitate_project/MNE-python_pipeline_v3/

if [ $# -ne 2 ];
    then echo "Please pass sub_prefix visit and step as command line arguments. E.g."
    echo "sbatch --array=103 srun_freesurfer.sh SA"
    echo "Exiting."
fi

module purge
module load FreeSurfer/6.0.1-centos6_x86_64; source ${FREESURFER_HOME}/SetUpFreeSurfer.sh

export SUBJECTS_DIR=/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/fs_defaced/

sub_prefix=$1       # Prefix of the subjects we're working on e.g. SA SB etc...
visit=$2
nifti_dir=/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/dicom2nifti/

set --

srun recon-all -i ${nifti_dir}sub-${sub_prefix}`printf "%03d" $SLURM_ARRAY_TASK_ID`/ses-VO/anat/${sub_prefix}`printf "%03d" $SLURM_ARRAY_TASK_ID`_MR_V0_anat_defaced.nii -s sub-${sub_prefix}`printf "%03d" $SLURM_ARRAY_TASK_ID` -all
