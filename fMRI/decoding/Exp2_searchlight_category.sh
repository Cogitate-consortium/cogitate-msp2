#!/bin/bash
# sbatch Exp2_searchlight_category.sh --subject=SD201 --condition=seen-seen_nogo --vg_or_replay=VG-Replay
# sbatch Exp2_searchlight_category.sh --subject=SD201 --condition=seen_nogo-seen --vg_or_replay=Replay-VG
#SBATCH --partition=xnat
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=zviroth@gmail.com
#SBATCH --time 12:00:00
#SBATCH --output=./slurm/slurm-category-%A_%a.out
#SBATCH --job-name=dcdCat

condition=""
subject=""
vg_or_replay=""
while [ $# -gt 0 ]; do
  case "$1" in
    --subject=*)
      subject="${1#*=}"
      ;;
    --condition=*)
      condition="${1#*=}"
      ;;  
    --vg_or_replay=*)
      vg_or_replay="${1#*=}"
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
  echo ${vg_or_replay}
done

cd /mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/code/Exp2/Zvi/slurm

module purge; module load Anaconda3/2020.11; source /hpc/shared/EasyBuild/apps/Anaconda3/2020.11/bin/activate; conda activate /hpc/users/$USER/.conda/envs/mne_ecog01

python3 ../Exp2_searchlight_category_forSlurm.py --subject "${subject}" --condition "${condition}" --vg_or_replay "${vg_or_replay}"