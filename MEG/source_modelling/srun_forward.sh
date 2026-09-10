#!/bin/bash
#SBATCH --partition=xnat
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=o.ferrante@bham.ac.uk
#SBATCH --time 5:00:00
#SBATCH --chdir=/hpc/users/oscar.ferrante/git/MEG_cogitate_project/MNE-python_pipeline_v3/

if [ $# -ne 3 ];
    then echo "Please pass sub_prefix visit and step as command line arguments. E.g."
    echo "sbatch --array=101,103,105 srun_forward.sh SA V1 surface"
    echo "Exiting."
fi

sub_prefix=$1       # Prefix of the subjects we're working on e.g. SA SB etc...
visit=$2
space=$3

set --

module purge
module load Anaconda3/2020.11
source /hpc/shared/EasyBuild/apps/Anaconda3/2020.11/bin/activate
conda activate /hpc/users/oscar.ferrante/.conda/envs/mne_meg01

srun python S01_forward_model.py --sub ${sub_prefix}`printf "%03d" $SLURM_ARRAY_TASK_ID` --visit ${visit} --space ${space}
