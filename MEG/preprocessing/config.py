# -*- coding: utf-8 -*-
"""
===========
Config file
===========

Configurate the parameters of the study.

"""

import os

# =============================================================================
# BIDS SETTINGS
# =============================================================================
bids_root = r'/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids'


# =============================================================================
# MAXWELL FILTERING SETTINGS
# =============================================================================

# Set filtering method
method='sss'
if method == 'tsss':
    st_duration = 10
else:
    st_duration = None


# =============================================================================
# FILTERING AND DOWNSAMPLING SETTINGS
# =============================================================================

# Filter and resampling params
l_freq = 1
h_freq = 40
sfreq = 100


# =============================================================================
# EPOCHING SETTINGS
# =============================================================================

# Set timewindow
tmin = -1
tmax = 2.5

# Epoch rejection criteria
reject_meg_eeg = dict(grad=4000e-13,    # T / m (gradiometers)
                      mag=6e-12        # T (magnetometers)
                      #eeg=200e-6       # V (EEG channels)
                      )
reject_meg = dict(grad=4000e-13,    # T / m (gradiometers)
                  mag=6e-12         # T (magnetometers)
                  )


# =============================================================================
# ICA SETTINGS
# =============================================================================

ica_method = 'fastica'
n_components = 0.99
max_iter = 800
random_state = 1688


# =============================================================================
# SOURCE MODELING
# =============================================================================

# Forward model
spacing='oct6'

# Inverse model
#   Beamforming
beam_method = 'dics'

active_win = (0.75, 1.25)
baseline_win = (-.5, 0)
