import sys
import os

import numpy as np

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
subject = subjects[int(sys.argv[1])]
sampling_rate = 100

project_dir = '/path/to/derivatives/'
data_dir = os.path.join(project_dir, 'preprocessing/sub-{}/ses-V2/meg'.format(subject))
fwd_dir = os.path.join(project_dir, 'forward/sub-{}/ses-V2/meg'.format(subject))
fs_dir = os.path.join(project_dir, 'fs')
power_dir = os.path.join(project_dir, 'ana1_save_power')
out_dir = os.path.join(project_dir, 'results')

n_perms = 10
n_pstrials = 3
test_ratio = .33

whitening = True

rdm_label = 'rdms'
sim_label = 'similarity'

# Output name encode whether whitening was applied.
if whitening:
    sim_label = sim_label + '_mnn'

similarity_file = os.path.join(out_dir, '{}_{}.pkl'.format(sim_label,subject))

