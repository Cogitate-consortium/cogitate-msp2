import os
import numpy as np
import pickle
import mne

import cfg


def load_data(file):
    """Load pickled object."""
    print('loading file: ' + file)
    with open(file, 'rb') as f:
        data = pickle.load(f)

    return(data)


def dump_data(data, filename):
    """Save pickled object."""
    print('writing file: ' + filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def get_data(data_file):
    """Load, preprocess, and cache MEG epochs for one task."""

    sub = data_file.split('_')[0].split('/')[-1]
    task = data_file.split('_')[-2].split('-')[-1]

    # Cache preprocessed epochs by subject, task, and sampling rate.
    ep_fname = 'epochs_{}_{}_sfreq-{}-epo.fif'.format(cfg.subject,task,cfg.sampling_rate)
    epoch_file = os.path.join(cfg.out_dir, ep_fname)

    if os.path.isfile(epoch_file):
        epochs = mne.read_epochs(epoch_file, preload=True, verbose=True).pick('meg')

    else:
        epochs = mne.read_epochs(data_file, preload=True, verbose=True).pick('meg')

        if task != 'rest':
            # Crop epochs.
            t_min = -.5
            t_max = 1.5
            epochs.crop(tmin=t_min, tmax=t_max,include_tmax=True, verbose=None)

        # Band-pass filter raw copy
        l_freq = .1
        h_freq = 40
        epochs.filter(l_freq, h_freq, n_jobs=-1)

        # Downsample
        sfreq = cfg.sampling_rate
        epochs.resample(sfreq, n_jobs=-1)

        if task != 'rest':
            # Baseline correction
            b_tmin = -.5
            b_tmax = -.3
            baseline = (b_tmin, b_tmax)
            epochs.apply_baseline(baseline=baseline)

        epochs.save(epoch_file)

    return(epochs)

def get_partition(dat, mdat, part, task):
    """Select a cognitive partition, optionally restricted in trial time."""
    # Remove blanks
    blank_str = 'Black' if task == 'replay' else 'Blank'
    valid_mask = mdat['Stimuli_type'] != blank_str
    dat = dat[valid_mask]
    mdat = mdat[valid_mask]

    # Parse partition
    if '-' in part and 'split' in part:
        cog_part, split = part.rsplit('-', 1)
        split_id = int(split.replace('split', ''))
    else:
        cog_part = part
        split_id = None

    # Cognitive partitioning
    if cog_part == 'all':
        mask = np.ones(len(mdat), dtype=bool)
    elif task == 'vg' and cog_part == 'seen':
        mask = mdat['Response'] == 'Seen'
    elif task == 'replay':
        if cog_part == 'seen-go':
            mask = mdat['Response'] == 'Seen'
        elif cog_part == 'seen-no-go':
            m1 = mdat['Trial_type'] == 'Non-Target'
            m2 = np.isnan(mdat['Response_time'])
            mask = np.logical_and(m1, m2)
        else:
            raise ValueError(f'Unknown replay partition: {cog_part}')
    else:
        raise ValueError(f'Unknown partition: {task}, {cog_part}')

    # Temporal partitioning
    if split_id is not None:
        bins = np.linspace(0, 1, cfg.n_splits + 1)
        low, high = bins[split_id - 1], bins[split_id]
        time_mask = (mdat['TrialProgress'] >= low) & (mdat['TrialProgress'] <= high)
        mask = mask & time_mask

    return dat[mask], mdat[mask]


def get_blanks(dat, mdat, task):
    """Select blank trials for the requested task."""

    if task == 'replay':
        blank_str = 'Black'
    elif task == 'vg':
        blank_str = 'Blank'

    blank_mask = mdat['Stimuli_type'] == blank_str
    dat = dat[blank_mask]
    mdat = mdat[blank_mask]

    return(dat, mdat)


def get_labels():
    """Return broad posterior and prefrontal anatomical label sets."""
    # Use fsaverage labels for the listed subjects.
    if cfg.subject in ['SA102', 'SA104', 'SA110', 'SA111', 'SA152']:
        labels = mne.read_labels_from_annot(subject="fsaverage",
                                                 parc='aparc.a2009s',
                                                 subjects_dir=cfg.fs_dir)
    else:
        labels = mne.read_labels_from_annot(subject=f"sub-{cfg.subject}",
                                                 parc='aparc.a2009s',
                                                 subjects_dir=cfg.fs_dir)

    pfc = ['G&S_cingul-Ant','G&S_cingul-Mid-Ant','G&S_cingul-Mid-Post','G_front_inf-Opercular',
        'G_front_inf-Orbital','G_front_inf-Triangul','G_front_middle','Lat_Fis-ant-Horizont',
        'Lat_Fis-ant-Vertical','S_front_inf','S_front_middle','S_front_sup']

    pos = [ 'G&S_occipital_inf','G_oc-temp_lat-fusifor','G_occipital_middle','G_cuneus',
        'G_occipital_sup','G_oc-temp_med-Lingual','G_oc-temp_med-Parahip','G_temporal_inf',
        'Pole_occipital','Pole_temporal','S_oc_middle&Lunatus','S_calcarine',
        'S_intrapariet&P_trans','S_oc_sup&transversal','S_temporal_sup']

    # Match atlas labels after removing hemisphere suffixes.
    labels_pfc = [x for x in labels if x.name[:-3] in pfc]
    labels_pos = [x for x in labels if x.name[:-3] in pos]

    rois = {'pfc': labels_pfc, 'pos': labels_pos}

    return(rois)


def get_labels_subset(subset):
    """Map requested ROI names to subject-specific anatomical labels."""

    # Read the same atlas used to define the requested ROI names.
    if cfg.subject in ['SA102', 'SA104', 'SA110', 'SA111', 'SA152']:
        labels = mne.read_labels_from_annot(subject="fsaverage",
                                                 parc='aparc.a2009s',
                                                 subjects_dir=cfg.fs_dir)
    else:
        labels = mne.read_labels_from_annot(subject=f"sub-{cfg.subject}",
                                                 parc='aparc.a2009s',
                                                 subjects_dir=cfg.fs_dir)

    pos = [ 'G&S_occipital_inf','G_oc-temp_lat-fusifor','G_occipital_middle','G_cuneus',
         'G_occipital_sup','G_oc-temp_med-Lingual','G_oc-temp_med-Parahip','G_temporal_inf',
         'Pole_occipital','Pole_temporal','S_oc_middle&Lunatus','S_calcarine',
         'S_intrapariet&P_trans','S_oc_sup&transversal','S_temporal_sup']
    labels_pos = [x for x in labels if x.name[:-3] in pos]

    # Preserve the caller's subset names in the returned mapping.
    rois = {}
    for sublist in subset:
        labels_subset = [x for x in labels_pos if x.name in subset[sublist]]
        rois[sublist] = labels_subset

    return(rois)


def source_data_roi(data, rank, baseline_cov, label_set, task):
    """Reconstruct source activity for a set of anatomical labels."""

    fname_fwd = os.path.join(cfg.fwd_dir, 'sub-{}_ses-V2_task-{}_surface_fwd.fif'.format(cfg.subject,task))
    fwd = mne.read_forward_solution(fname_fwd)

    # Make inverse operator
    inv = mne.minimum_norm.make_inverse_operator(data.info, fwd, baseline_cov, depth=.8, fixed=True, rank=rank, use_cps=True)

    snr = 3.0
    lambda2 = 1.0 / snr ** 2

    src_dat = []
    n_vertices = []
    vertices = []

    # Reconstruct each label separately to retain vertex bookkeeping.
    for label in label_set:

        print(label)
        stcs = mne.minimum_norm.apply_inverse_epochs(data, inv, lambda2, 'MNE', pick_ori=None, label=label)
        d = np.array([x.data for x in stcs])
        src_dat.append(d)
        n_vertices.append(d.shape[1])

        verts_label = np.concatenate(stcs[0].vertices)
        assert len(verts_label) == d.shape[1]
        vertices.append(verts_label)

    # Concatenate labels along the source-vertex axis.
    src_data = np.concatenate(src_dat, axis=1)
    vertices = np.concatenate(vertices)

    return(src_data, n_vertices, vertices)


def get_data_rois_subset(task, rois_list):
    """Load epochs and reconstruct activity for selected ROI subsets."""

    data_file = os.path.join(cfg.data_dir, 'sub-{}_ses-V2_task-{}_epo.fif'.format(cfg.subject,task))
    rest_file = os.path.join(cfg.data_dir, 'sub-{}_ses-V2_task-rest_epo.fif'.format(cfg.subject))

    # Load task and rest epochs.
    data = get_data(data_file)
    rest = get_data(rest_file)

    tmin_data = data.times[0]
    tmax_data = data.times[-1]

    tmin_rest = rest.times[0]
    tmax_rest = rest.times[-1]

    metadata = data.metadata

    # Use the lower task/rest MEG rank for covariance and inversion.
    rank = mne.compute_rank(data, tol=1e-6, tol_kind='relative')
    rank_rest = mne.compute_rank(rest, tol=1e-6, tol_kind='relative')
    if rank_rest['meg']<rank['meg']:
        rank=rank_rest

    baseline_cov = mne.compute_covariance(rest, tmin=tmin_rest, tmax=tmax_rest, method='empirical', rank=rank, n_jobs=-1, verbose=True)

    # Use replay partitions or all VG epochs for active covariance.
    if task == 'replay':
            dats = []
            for part in cfg.task_partitions[task]:
                d, _ = get_partition(data, metadata, part, task)
                dats.append(d)

            active_cov = mne.compute_covariance(dats, tmin=tmin_data, tmax=tmax_data, method='empirical', rank=rank, n_jobs=-1, verbose=True)

    else:
        active_cov = mne.compute_covariance(data, tmin=tmin_data, tmax=tmax_data, method='empirical', rank=rank, n_jobs=-1, verbose=True)

    # Combine rest and active covariance for source reconstruction.
    common_cov = baseline_cov + active_cov

    roi_labels = get_labels_subset(rois_list)

    # Reconstruct each ROI subset.
    data_rois = {x: source_data_roi(data, rank, common_cov, roi_labels[x], task) for x in roi_labels}

    return(data_rois, metadata)


def get_data_roi_posterior(task):
    """Load epochs and reconstruct activity for posterior cortical labels."""

    data_file = os.path.join(cfg.data_dir, 'sub-{}_ses-V2_task-{}_epo.fif'.format(cfg.subject,task))
    rest_file = os.path.join(cfg.data_dir, 'sub-{}_ses-V2_task-rest_epo.fif'.format(cfg.subject))
    out_file = os.path.join(cfg.out_dir, 'src_data_roi_post_{}_{}.pkl'.format(task, cfg.subject))

    # Cache the result.
    if os.path.isfile(out_file):
        out = load_data(out_file)
        return(out['data_roi'], out['metadata'], out['n_vertices'], out['vertices'], out['times'])

    else:
        data = get_data(data_file)
        rest = get_data(rest_file)

        tmin_rest = rest.times[0]
        tmax_rest = rest.times[-1]

        metadata = data.metadata

        # Projecting sensor-space data to source space 
        rank = mne.compute_rank(data, tol=1e-6, tol_kind='relative')
        rank_rest = mne.compute_rank(rest, tol=1e-6, tol_kind='relative')

        if rank_rest['meg']<rank['meg']:
            rank=rank_rest
        baseline_cov = mne.compute_covariance(rest, tmin=tmin_rest, tmax=tmax_rest, method='empirical', rank=rank, n_jobs=-1, verbose=True)
        active_cov = mne.compute_covariance(data, tmin=-.5, tmax=-.25, method='empirical', rank=rank, n_jobs=-1, verbose=True)

        n1 = baseline_cov['nfree']
        n2 = active_cov['nfree']

        common_cov = baseline_cov.copy()
        common_cov['data'] = (baseline_cov['data'] * n1 + active_cov['data'] * n2) / (n1 + n2)
        common_cov['nfree'] = n1 + n2

        # Reconstruct all posterior labels and cache them as one vertex axis.
        roi_labels = get_labels()
        data_roi, n_vertices, vertices = source_data_roi(data, rank, common_cov, roi_labels['pos'], task)

        out = {'data_roi': data_roi, 'metadata': metadata, 'n_vertices': n_vertices, 'vertices': vertices, 'times': data.times}
        dump_data(out, out_file)

        return(data_roi, metadata, n_vertices, vertices, data.times)


def get_cdata(dat, mdata, target):
    """Split trials into category or hemifield conditions."""

    mdat = mdata.copy()

    # location labels to hemifield
    if target == 'location':
        tcol = 'Location'
        locs = mdat[tcol].astype(str)
        side = np.array([x.split(' ')[1] for x in locs])
        mdat[tcol] = side
    elif target == 'category':
        tcol = 'Stimuli_type'

    conds, conds_count = np.unique(mdat[tcol].astype(str), return_counts=True)

    if len(conds) != 2:
        print(f'no analysis possible for {target}; found conditions {conds} with counts {conds_count}')
        return None, None, None

    # Return one trial array per sorted condition label.
    cdat = [dat[mdat[tcol].values == x] for x in conds]

    return(cdat, conds, conds_count)

