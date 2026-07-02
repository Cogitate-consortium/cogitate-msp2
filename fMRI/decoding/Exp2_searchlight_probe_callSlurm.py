"""
Author: Zvi Roth
Date created: 23-06-2024
Purpose: Creates Slurm command to run analysis over group of subjects
Condition: seen or unseen or all , or seen-unseen, or unseen-unseen, etc. (Runs only on VG which has probes)
Command: python3 Exp2_searchlight_probe_callSlurm.py --condition "seen"
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

# use code for comparing category/location decoding in seen vs. unseen:
analysis_check_data = csv_data.DECODING_min_sl_sr_ul_ur
analysis_check_list = analysis_check_data.tolist()

def run_analysis():
    # Parse command line inputs:
    parser = argparse.ArgumentParser(
        description="Implements fMRI searchlight analysis for experiment2")
    parser.add_argument('--condition', type=str, default=None,
                        help="seen, unseen, seen-seen, unseen-senn, unseen-unseen, etc.")

    args = parser.parse_args()
    condition = args.condition

    print('starting to loop over subjects')
    print(subject_list)
    # subject counter
    i=0
    # Loop over subjects
    for sub_code, analysis_check in zip(subject_list, analysis_check_list):

        #REMOVE FOR ACTUAL SUBJECT LIST!!
        analysis_check = 'TRUE'

        if analysis_check == 'TRUE':
            #slurm command
            run_command = f"sbatch Exp2_searchlight_probe.sh --subject={sub_code} --condition={condition} "
            subprocess.Popen(run_command, shell=True)
            i = i + 1

            print('finished sub ' + str(i) + ': ' + sub_code)
        else:
            print('excluding subject ' + sub_code)
    toc = time.time()
    print('total runtime: ' + str(toc-tic))

if __name__ == "__main__":
  run_analysis()
toc = time.time()
