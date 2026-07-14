import sys
import os

import numpy as np

# Select local or HPC paths from the operating system.
if sys.platform == 'darwin':
    hpc = False
    dummy = True
elif sys.platform == 'win32':
    hpc = False
    dummy = True
else:
    hpc = True
    dummy = False


# Subjects and experimental conditions.
subjects = ['SA101','SA104','SA108','SA111','SA112','SA114','SA116','SA118','SA121','SA123','SA124',
            'SA126','SA127','SA131','SA132','SA136','SA138','SA139','SA140','SA144','SA145',
            'SA146','SA148','SA150','SA151','SA154','SA160','SA166','SA170','SA173','SA174',
            'SA176','SB006','SB008','SB009','SB011','SB013','SB015','SB016','SB019','SB023','SB024',
            'SB028','SB030','SB031','SB036','SB039','SB040','SB041','SB042','SB044','SB049','SB050',
            'SB051','SB061','SB063','SB069','SB071','SB073','SB074','SB081','SB084','SB099']

tasks = ['vg', 'replay']
partitions = ['all', 'non_target', 'seen']

task_partitions = {'vg': ['seen'],
                    'replay': ['seen-go', 'seen-no-go']}

content_targets = ['category', 'location']

# Subject-specific paths and runtime settings.
if hpc:
    subject = subjects[int(sys.argv[1])]
    sampling_rate = 100
    data_dir = '/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/preprocessing/sub-{}/ses-V2/meg'.format(subject)
    fwd_dir = '/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/forward/sub-{}/ses-V2/meg'.format(subject)
    fs_dir = '/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/fs'
    power_dir = '/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/ana1_save_power'
    out_dir = '/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/rsa_test1/results'

else:
    subject = 'SA148'
    sampling_rate = 100
    data_dir = '/Users/pablo/Documents/phd/cogitate/meg/data'
    fwd_dir = os.path.join(data_dir, 'forward')
    fs_dir = os.path.join(data_dir, 'fs')
    power_dir = '/Users/pablo/Documents/phd/cogitate/meg/data/ana1_save_power'
    out_dir = '/Users/pablo/Documents/phd/cogitate/meg/results'


# Lightweight local settings versus full cluster settings.
if dummy:
    n_perms = 1
else:
    n_perms = 10

n_pstrials = 3
test_ratio = .33

whitening = True

dec_label = 'decoding'
rdm_label = 'rdms'
sim_label = 'similarity'

# Output names encode whether whitening was applied.
if whitening:
    dec_label = dec_label + '_mnn'
    rdm_label = rdm_label + '_mnn'
    sim_label = sim_label + '_mnn'

decoding_file = os.path.join(out_dir, '{}_{}.pkl'.format(dec_label,subject))

similarity_file_source = os.path.join(out_dir, '{}_rois_{}.pkl'.format(sim_label,subject))
similarity_file_source_subroi = os.path.join(out_dir, '{}_rois_subset_{}.pkl'.format(sim_label,subject))
similarity_file_source_single = os.path.join(out_dir, '{}_rois_single_{}.pkl'.format(sim_label,subject))

rdm_file = os.path.join(out_dir, '{}_{}.pkl'.format(rdm_label,subject))
