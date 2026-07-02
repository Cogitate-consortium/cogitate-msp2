"""
Author: Zvi Roth
Date created: 23-06-2024
Purpose: Creates Slurm command to run analysis over group of subjects
vg_or_replay: VG or Replay, or VG-Replay, or VG-VG, etc.
Condition: seen or unseen, or seen-unseen, or unseen-unseen, etc.
Command: python3 Exp2_searchlight_probe_group_callSlurm.py --condition "seen"
"""

import os
import pandas as pd
import time
import subprocess
import argparse

def run_analysis():

    # Parse command line inputs:
    parser = argparse.ArgumentParser(
        description="Implements fMRI searchlight analysis for experiment2")
    parser.add_argument('--condition', type=str, default=None,
                        help="seen, unseen, seen-seen, unseen-senn, unseen-unseen, etc.")

    args = parser.parse_args()
    condition = args.condition

    run_command = f"sbatch Exp2_searchlight_probe_group.sh --condition={condition}"
    subprocess.Popen(run_command, shell=True)


tic = time.time()
if __name__ == "__main__":
  run_analysis()
toc = time.time()
print('total runtime: ' + str(toc - tic))

