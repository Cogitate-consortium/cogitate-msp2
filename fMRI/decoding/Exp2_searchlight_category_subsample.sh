#!/bin/bash
# sbatch Exp2_searchlight_location_subsample.sh --subject=SD201 --condition=seen
#SBATCH --partition=xnat
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=zviroth@gmail.com
#SBATCH --time 6-12:00:00
#SBATCH --output=./slurm/slurm-catSample-%A_%a.out
#SBATCH --job-name=catSamp

condition=""
subject=""
while [ $# -gt 0 ]; do
  case "$1" in
    --subject=*)
      subject="${1#*=}"
      ;;
    --condition=*)
      condition="${1#*=}"
      ;;  
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument: ${1}*\n"
      printf "***************************\n"
      exit 1

  esac
  shift
  echo ${subject}
  echo ${condition}
done

cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Exp2/Zvi/slurm

module purge; module load Anaconda3/2020.11; source /hpc/shared/EasyBuild/apps/Anaconda3/2020.11/bin/activate; conda activate /hpc/users/$USER/.conda/envs/mne_ecog01

python3 ../Exp2_searchlight_category_subsample_forSlurm.py --subject "${subject}" --condition "${condition}"