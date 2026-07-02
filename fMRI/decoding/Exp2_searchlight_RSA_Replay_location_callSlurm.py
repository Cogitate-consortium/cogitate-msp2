"""
Author: Zvi Roth
Date created: 23-06-2024
Purpose: Creates Slurm command to run analysis over group of subjects
vg_or_replay: VG or Replay, or VG-Replay, or VG-VG, etc.
Condition: seen or unseen, or seen-unseen, or unseen-unseen, etc.
Command: python3 Exp2_searchlight_RSA_Replay_location_callSlurm.py
"""

import os
import pandas as pd
import time
import subprocess
import argparse

tic=time.time()


# CHANGE FOR ACTUAL SUBJECTS!!
#csv_file_end = 'ses-v2-optimization-subs.csv'
csv_file_end = 'ses-v2-analysis-subs.csv'

csv_file = os.path.join('/mnt/beegfs/XNAT/COGITATE/fMRI/phase_2/processed/bids/derivatives/qcs/', csv_file_end)
csv_data= pd.read_csv(csv_file, sep=',')
subjects_phase2 = csv_data.sub_code
subject_list=subjects_phase2.tolist()

# use code for RSA:
analysis_check_data = csv_data.DECODING_min_sl_sr
analysis_check_list = analysis_check_data.tolist()

def run_analysis():
    print('starting to loop over subjects')
    print(subject_list)
    # subject counter
    i = 0
    # Loop over subjects
    for sub_code, analysis_check in zip(subject_list, analysis_check_list):

        #REMOVE FOR ACTUAL SUBJECT LIST!!
        #analysis_check = 'TRUE'


        if analysis_check:
            #slurm command
            run_command = f"sbatch Exp2_searchlight_RSA_Replay_location.sh --subject={sub_code}"
            subprocess.Popen(run_command, shell=True)
            i = i + 1

            print('finished sub ' + str(i) + ': ' + sub_code)
        else:
            print('excluding subject ' + sub_code)
    toc = time.time()
    print('subject list: DECODING_min_sl_sr')
    print('total runtime: ' + str(toc-tic))


if __name__ == "__main__":
  run_analysis()
toc = time.time()
