# AT-seen-go — no scheduler (Wish)

Local copy of `gPPI_code/AT-seen-go/` for hosts **without Slurm**. Gitignored; do not commit.

Science, EVs, FSF templates, and BIDS **outputs** match the Slurm pipeline. FEAT maps still go to:

`{BIDS_ROOT}/derivatives/fslFeat/gPPI_AT-seen-go/`

`fsf_files/` and `job_files/` stay in **this** folder.

## Setup

FSL (`feat`, `fslmeants`, `fslhd`, `fslstats`) must already be on `PATH`. There is no `module load` requirement.

```bash
export BIDS_ROOT=/wishData/cogitate-data/fMRI/bids   # adjust
export CODE_PATH="$HOME/cogitate-msp2/fMRI"          # this clone
export GPPI_NJOBS=8                                  # optional; default 8
```

Delete a job’s `.run` marker under `job_files/` to retry that step.

## Run order

```bash
cd "${CODE_PATH}/gPPI_code/no_slurm/AT-seen-go"

bash 01_create_ev_files.sh
bash 02_make_timecourse_files.sh
bash 03_make_fsf_files.sh
bash 04_run_PPI_1st_level.sh
bash 05_create_identity_reg_for_2nd_lvl.sh
bash 06_make_fsf_files_2ndlevel.sh
bash 07_run_PPI_2nd_level_Replay.sh
bash 08_create_identity_reg_for_3rd_lvl.sh
bash 09_make_fsf_files_3rdlevel.sh
bash 10_run_PPI_3rd_level.sh
bash run_average_gppi_in_evc.sh
python3 12_gppi_group_analysis_BF.py
python3 13_gppi_group_level_tables_BF.py

bash make_3rd_level_inclusion_report.sh
bash inspect_2nd_level_for_3rd.sh
bash run_copy_gppi_2nd_lvl_for_evc.sh
```

02 / 04 / 07 / 10 run jobs in parallel with `xargs -P "$GPPI_NJOBS"` (`FSLPARALLEL=0` so FEAT does not call `fsl_sub`).
