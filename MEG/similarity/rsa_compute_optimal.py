import os
import mne
import numpy as np
import pickle

import cfg
from utils import *
from rsa_compute_core import *


def get_metrics(rp_input, vg_input):
    """Vertex-wise differences used for selection."""
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

    bin_start = .3
    bin_size = .4
    task = 'replay'
    part = 'seen-no-go'

    print('Processing task: {}'.format(task))
    out = {'psd_blank': [], 'psd_stim':[], 't_ind':[], 'p_ind':[]}
    out['task'] = task
    out['freqs'] = None
    out['gamma_band'] = {'fmin': 35, 'fmax': 90}

    data_rp, mdata_rp, n_vertices, vertices, times = rp_input
    data_vg, mdata_vg, _, _, _ = vg_input
    V = sum(n_vertices)

    out['mdata_rp'] = mdata_rp
    out['mdata_vg'] = mdata_vg
    out['vertices'] = vertices
    out['times'] = times

    # Vertex selection is estimated in the 0.3-0.7 s window.
    mask = (times >= bin_start) & (times <= bin_start + bin_size)
    datw_rp = data_rp[:, :, mask]
    datw_vg = data_vg[:, :, mask]
    n_time = np.sum(mask)

    # Keep blank and no-go spectra as diagnostics for the selection window.
    blank, mblank = get_blanks(datw_rp, mdata_rp, 'replay')
    psd_blank, freqs = get_spectrum(blank, fs=cfg.sampling_rate)
    out['psd_blank'].append(psd_blank.mean(axis=0))
    out['freqs'] = freqs

    # Assemble the three task/response partitions used for selection.
    stim_rpng, mstim_rpng = get_partition(datw_rp, mdata_rp, 'seen-no-go', 'replay')
    psd_stim, _ = get_spectrum(stim_rpng, fs=cfg.sampling_rate)
    out['psd_stim'].append(psd_stim.mean(axis=0))
    stim_rpg, mstim_rpg = get_partition(datw_rp, mdata_rp, 'seen-go', 'replay')
    stim_vg, mstim_vg = get_partition(datw_vg, mdata_vg, 'seen', 'vg')

    # Split each partition by category, then average trials.
    c_rpg, conds, rpg_count  = get_cdata(stim_rpg, mstim_rpg, 'category')
    c_rpng, _, rpng_count  = get_cdata(stim_rpng, mstim_rpng, 'category')
    c_vg, _, vg_count  = get_cdata(stim_vg, mstim_vg, 'category')

    m_rpg = np.array([c_rpg[x].mean(axis=0) for x in range(len(conds))])
    m_rpng = np.array([c_rpng[x].mean(axis=0) for x in range(len(conds))])
    m_vg = np.array([c_vg[x].mean(axis=0) for x in range(len(conds))])

    n_conds = len(conds)
    sim = np.full((n_conds, V, n_time), np.nan, dtype=np.float32)
    tsk = []
    cnt = []
    # Compare categories within partitions and matched categories across partitions.
    for i, da in enumerate((m_vg, m_rpg, m_rpng)):
        for j, db in enumerate((m_vg, m_rpg, m_rpng)):
            if j < i:
                continue
            elif i == j:
                tsk.append(np.abs(da[0] - db[1]))
            else:
                cnt.append(np.abs(da[0] - db[0]))
                cnt.append(np.abs(da[1] - db[1]))

    out['task'] = np.mean(tsk, axis=0)
    out['content'] = np.mean(cnt, axis=0)

    return(out)


def cv_loader(in_tuple, test_ratio):
    """Stratified train/test splits while preserving source metadata."""
    from sklearn.model_selection import StratifiedShuffleSplit

    dat = in_tuple[0]
    mdat = in_tuple[1]

    stim = mdat['Stimuli_type'].astype(str).values
    resp = mdat['Response'].astype(str).values
    
    # Stratify on the joint stimulus-response label.
    strat_key = np.array([f'{s}_{r}' for s, r in zip(stim, resp)])
    strat_key[strat_key == 'Blank_Seen'] = 'Blank_None'
    strat_key[strat_key == 'Object_Unseen'] = 'invalid'
    strat_key[strat_key == 'Face_Unseen'] = 'invalid'

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio)
    train_idx, test_idx = next(sss.split(dat, strat_key))

    def slice_data(idx):
        """Slice data and metadata while retaining source-space descriptors."""
        return (dat[idx], mdat.iloc[idx].reset_index(drop=True), 
                in_tuple[2], in_tuple[3], in_tuple[4])

    return {'train': slice_data(train_idx), 'test': slice_data(test_idx)}


def get_allsubs_metrics(iterations=10, test_ratio=.5):
    """Cross-validate vertex selection and RSA across all subjects."""
    import time

    failed = []
    timings = {}
    for subject in cfg.subjects:
        start_time = time.perf_counter()

        # Repoint the shared configuration to the current subject.
        cfg.meg_dir = os.path.join(cfg.data_dir, 'preprocessing', f'sub-{subject}', 'ses-V2', 'meg')
        cfg.fwd_dir = os.path.join(cfg.data_dir, 'forward', f'sub-{subject}', 'ses-V2', 'meg')
        cfg.fs_dir = os.path.join(cfg.data_dir, 'fs')
        cfg.out_dir = '/Volumes/pablo_cfin/cogitate_data/results'
        cfg.subject = subject

        try:
            # Load full posterior source data before cross-validation.
            rp_tuple = get_data_roi_posterior('replay')
            vg_tuple = get_data_roi_posterior('vg')

            for iteration in range(iterations):
                print(f'>>Subject: {cfg.subject} - CV iteration: {iteration}')
                # Estimate vertex sensitivity on the training split only.
                rp_split = cv_loader(rp_tuple, test_ratio)
                vg_split = cv_loader(vg_tuple, test_ratio)
                metrics = get_metrics(rp_split['train'], vg_split['train'])
                ratio = (metrics['task'] / metrics['content']).mean(axis=1)

                # Partition the held-out split for unbiased similarity evaluation.
                data, metadata = {}, {}
                for task, dtask in zip(cfg.tasks,(vg_split, rp_split)):
                    data[task], metadata[task] = {}, {}
                    d, md, n_vertices, vertices, times = dtask['test']
                    for part in cfg.task_partitions[task]:
                        print(f'>>loading data:{cfg.subject}-{task}-{part}')
                        data[task][part], metadata[task][part] = get_partition(d, md, part, task)

                # Retain vertices with the largest task/content metric ratio.
                for percentile in [90, 95]:
                    thr = np.percentile(ratio, percentile)
                    mask = ratio >= thr

                    # Apply the training-derived mask and downsample test data.
                    data_opt = {}
                    for task in cfg.tasks:
                        data_opt[task] = {}
                        for part in cfg.task_partitions[task]:
                            data_opt[task][part] = data[task][part][:, mask, ::10]

                    print(f'>>RSA: {cfg.subject} - p{percentile}')
                    rsa_results = rsa_core(data_opt, metadata)

                    out_file = os.path.join(cfg.out_dir, f'rsa_optimal_p{percentile}_{cfg.subject}_cvi{iteration}.pkl')
                    dump_data(rsa_results, out_file)
                end_time = time.perf_counter()
                timings[subject] = end_time - start_time

        except Exception as e:
            print(f"Subject {subject} failed: {e}")
            failed.append(subject)
            continue


if __name__ == '__main__':
    get_allsubs_metrics()

