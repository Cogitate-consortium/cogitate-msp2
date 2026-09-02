import sys
import os
import mne
import numpy as np

import cfg as cfg
from utils import *
from rsa_compute_core import *


def get_metrics(hilbert_gamma=False):
    """Estimate stimulus-versus-blank spectral and distance metrics."""
    from scipy.stats import ttest_ind, ttest_1samp
    from mne.filter import filter_data
    from scipy.signal import hilbert

    def get_gamma_envelope(dat, fs, fmin=35, fmax=90):
        """Hilbert amplitude envelope in the gamma band."""
        N, V, T = dat.shape
        x = dat.reshape(-1, T)
        # narrowband filter
        x_filt = filter_data(x, sfreq=fs, l_freq=fmin, h_freq=fmax, method='fir', n_jobs=3, verbose=True)
        x_filt = x_filt.reshape(N, V, T)
        # analytic signal + envelope
        analytic = hilbert(x_filt, axis=-1)
        env = np.abs(analytic)
        return env

    def get_spectrum(dat, fs=500):
        """Multitaper power spectra for each trial and vertex."""
        from mne.time_frequency import psd_array_multitaper

        Nstim, V, B = dat.shape
        data = dat.reshape(-1, B)
        psd_stim, freqs = psd_array_multitaper(data,sfreq=fs,fmin=0,fmax=100,
                                               adaptive=True,low_bias=True,
                                               normalization='full',n_jobs=-1)
        n_freqs = psd_stim.shape[-1]
        psd_stim = psd_stim.reshape(Nstim, V, n_freqs)
        return(psd_stim, freqs)

    def cosine_distance(a, b):
        """Cosine distance per vertex."""
        num   = np.sum(a * b, axis=1)
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        dist = 1 - (num / denom)
        return dist

    def euclidean_distance(a, b):
        """Euclidean distance per vertex."""
        dist = np.linalg.norm(a - b, axis=1)
        return dist

    def distance_test(stim, blank, metric, n_perm=1000, seed=None):
        """Test stimulus-blank distance against shuffled group labels."""
        rng = np.random.default_rng(seed)
        if metric == 'euclidean':
            dist_func = euclidean_distance
        elif metric == 'cosine':
            dist_func = cosine_distance
        else:
            raise ValueError('Unknown metric: {}'.format(metric))   
        # Observed Euclidean distance per vertex
        dist_obs = dist_func(stim.mean(axis=0), blank.mean(axis=0))  # (V,)
        # Permutation null
        Nstim = stim.shape[0]
        Nblank = blank.shape[0]
        Ntotal = Nstim + Nblank
        combined = np.concatenate([stim, blank], axis=0)  # (Ntotal, V, B)
        null = np.zeros((n_perm, dist_obs.size))
        for p in range(n_perm):
            idx = rng.permutation(Ntotal)
            stim_p = combined[idx[:Nstim]]
            blank_p = combined[idx[Nstim:]]
            null[p] = dist_func(stim_p.mean(axis=0), blank_p.mean(axis=0))
        # p-values (greater tail)
        pvals = (np.sum(null >= dist_obs[None, :], axis=0) + 1) / (n_perm + 1)
        return dist_obs, pvals

    tbins = [-.25, .3]
    bin_size = .5
    out_file = os.path.join(cfg.out_dir, 'psd_stat_{}_{}bins.pkl'.format(cfg.subject, len(tbins)))

    if os.path.isfile(out_file):
        out = load_data(out_file)
        return(out) 

    else:        
        out = {}
        for task in cfg.tasks:
            # Load posterior source activity and its vertex bookkeeping.
            print('Processing task: {}'.format(task))
            out[task] = {}
            out[task]['freqs'] = None

            data_roi, metadata, n_vertices, vertices, times = get_data_roi_posterior(task)
            V = sum(n_vertices)

            out[task]['metadata'] = metadata
            out[task]['vertices'] = vertices
            out[task]['times'] = times

            # Hilbert gamma is optional; multitaper PSD is always computed.
            if hilbert_gamma:
                gamma_env = get_gamma_envelope(data_roi, fs=cfg.sampling_rate, fmin=35, fmax=90)

            out[task]['gamma_band'] = {'fmin': 35, 'fmax': 90}
            out[task]['psd_blank'] = []
            out[task]['gamma_blank'] = []

            # Each partition stores one metric vector per analysis window.
            for part in cfg.task_partitions[task]:
                out[task][part] = {'psd_stim':[], 't_ind':[], 'p_ind':[], 
                                                    't_dif':[], 'p_dif':[], 
                                                    'euc':[], 'p_euc':[], 
                                                    'cos':[], 'p_cos':[],
                                                    'gamma_stim': [],
                                                    't_gamma': [], 'p_gamma': []}

            # Compare stimulus and blank activity within each time window.
            for b in tbins:
                mask = (times >= b) & (times <= b + bin_size)
                datw = data_roi[:, :, mask]

                if hilbert_gamma:
                    gammaw = gamma_env[:, :, mask]
                    gamma_blank, _ = get_blanks(gammaw, metadata, task)
                    gamma_blank_mean = gamma_blank.mean(axis=2)
                    out[task]['gamma_blank'].append(gamma_blank_mean)

                # Blank trials define the reference distribution.
                blank, mblank = get_blanks(datw, metadata, task)
                blank_rms = np.sqrt((blank**2).mean(axis=2))
                psd_blank, freqs = get_spectrum(blank, fs=cfg.sampling_rate)
                out[task]['psd_blank'].append(psd_blank.mean(axis=0))

                if out[task]['freqs'] is None:
                    out[task]['freqs'] = freqs

                for part in cfg.task_partitions[task]:
                    # Extract the stimulus partition matched to this task.
                    stim, mstim = get_partition(datw, metadata, part, task)
                    stim_rms  = np.sqrt((stim**2).mean(axis=2))

                    if hilbert_gamma:
                        # Compare mean gamma envelopes between stimulus and blank trials.
                        gamma_stim, _ = get_partition(gammaw, metadata, part, task)
                        gamma_stim_mean = gamma_stim.mean(axis=2)
                        out[task][part]['gamma_stim'].append(gamma_stim_mean)

                        t_gamma, p_gamma = ttest_ind(gamma_stim_mean, gamma_blank_mean, axis=0, equal_var=False)
                        out[task][part]['t_gamma'].append(t_gamma)
                        out[task][part]['p_gamma'].append(p_gamma)

                    # Store mean spectra, then test broadband RMS per vertex.
                    psd_stim, _ = get_spectrum(stim, fs=cfg.sampling_rate)
                    out[task][part]['psd_stim'].append(psd_stim.mean(axis=0))

                    t_ind, p_ind = ttest_ind(stim_rms, blank_rms, axis=0, equal_var=False)
                    out[task][part]['t_ind'].append(t_ind)
                    out[task][part]['p_ind'].append(p_ind)

                    # Test differences over time.
                    stim_mean = stim.mean(axis=0)
                    blank_mean = blank.mean(axis=0)
                    abs_dif = np.abs(stim_mean - blank_mean)
                    t_dif, p_dif = ttest_1samp(abs_dif, popmean=0, axis=1, alternative='greater')

                    out[task][part]['t_dif'].append(t_dif)
                    out[task][part]['p_dif'].append(p_dif)

                    # Complement the univariate tests with multivariate distances.
                    euc, p_euc = distance_test(stim, blank, 'euclidean')
                    out[task][part]['euc'].append(euc)
                    out[task][part]['p_euc'].append(p_euc)

                    cos, p_cos = distance_test(stim, blank, 'cosine')
                    out[task][part]['cos'].append(cos)
                    out[task][part]['p_cos'].append(p_cos)

        dump_data(out, out_file)

        return(out)


def get_singlesub_metrics(subject):
    """Select vertices by gamma activity and run subject-level similarity."""
    import time

    cfg.sampling_rate = 500
    bin = 1
    metrics = get_metrics(hilbert_gamma=False)

    # Load source data and preselect each cognitive partition.
    data, metadata = {}, {}
    for task in cfg.tasks:
        data[task], metadata[task] = {}, {}
        d, md, n_vertices, vertices, times = get_data_roi_posterior(task)

        for part in cfg.task_partitions[task]:
            print(f'>>loading data:{cfg.subject}-{task}-{part}')
            data[task][part], metadata[task][part] = get_partition(d, md, part, task)

    # Convert the selected gamma estimator into one value per vertex.
    for solution in ['multitaper']:
        if solution == 'hilbert':
            gamma_stim = np.array(metrics['replay']['seen-no-go']['gamma_stim']).mean(axis=1)
            gamma_blank = np.array(metrics['replay']['gamma_blank']).mean(axis=1)

        elif solution == 'multitaper':
            freq_mask = metrics['replay']['freqs'] >= 35
            gamma_stim = np.array(metrics['replay']['seen-no-go']['psd_stim'])[:,:,freq_mask]
            gamma_blank = np.array(metrics['replay']['psd_blank'])[:,:,freq_mask]

        for comparison in ['dif']:
            # Use either stimulus or its log ratio to blank.
            if comparison == 'stim':
                gamma_ = gamma_stim[bin]
            elif comparison == 'dif':
                gamma_ = np.log10(gamma_stim[bin] / gamma_blank[bin])

            for selection in ['1d', '2d']:
                # Collapse the gamma-frequency axis by its mean or maximum.
                if selection=='1d':
                    gamma = np.mean(gamma_, axis=1)
                elif selection=='2d':
                    gamma = np.max(gamma_, axis=1)

                for percentile in [95, 90, -90, -95]:
                    # Negative percentiles are size-matched random controls.
                    if percentile < 0:
                        N = len(gamma)
                        frac = abs(percentile) / 100
                        n_select = int((1 - frac) * N)
                        idx = np.random.choice(N, size=n_select, replace=False)
                        mask = np.zeros(N, dtype=bool)
                        mask[idx] = True
                    else:
                        thr = np.percentile(gamma, percentile)
                        mask = gamma >= thr

                    # Restrict vertices and downsample from 500 to 50 Hz.
                    data_gamma = {}
                    for task in cfg.tasks:
                        data_gamma[task] = {}
                        for part in cfg.task_partitions[task]:
                            data_gamma[task][part] = data[task][part][:, mask, ::10]

                    print(f'>>>>>RSA: {cfg.subject} - {solution} - {comparison} - p{percentile}')
                    rsa_results = rsa_core(data_gamma, metadata)

                    out_file = os.path.join(cfg.out_dir, f'rsa-gamma_{solution}_{comparison}_{selection}_p{percentile}_bin{bin}_subroi_{cfg.subject}.pkl')
                    dump_data(rsa_results, out_file)


if __name__ == '__main__':
    get_singlesub_metrics(cfg.subjects[int(sys.argv[1])])



