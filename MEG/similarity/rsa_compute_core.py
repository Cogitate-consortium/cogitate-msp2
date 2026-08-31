import os
import sys
import numpy as np

import scipy
from sklearn.covariance import LedoitWolf

import cfg
from utils import *


def get_rois_list():
    """Define posterior ROI subsets from alpha and gamma power differences."""
    fname_freq = 'power_dif_rois_{}.pkl'.format(cfg.subject)
    sfile_freq = os.path.join(cfg.out_dir, fname_freq)
    freq_data = load_data(sfile_freq)

    win = [49, 99]  # Time window

    # Average each ROI/frequency over the selected window.
    dat = {y: {z: np.nanmean(freq_data['pos'][y][z][:, win[0]:win[1]], axis=(0, 1))
            for z in freq_data['pos'][y]}
        for y in freq_data['pos']}

    rois = {}
    for n_rois in [6]:
        # Keep the strongest and weakest ROIs in each frequency band.
        for freq in ['alpha', 'gamma']:
            label = f'{n_rois}_{freq}'
            fmax = sorted(dat.items(), key=lambda x: x[1][freq], reverse=True)[:n_rois]
            fmin = sorted(dat.items(), key=lambda x: x[1][freq])[:n_rois]
            rois[label + '_max'] = [x[0] for x in fmax]
            rois[label + '_min'] = [x[0] for x in fmin]

        # Main subset: alpha minima combined with gamma maxima.
        label_combi = f'combined_{n_rois}'
        combi = rois[f'{n_rois}_alpha_min'] + rois[f'{n_rois}_gamma_max']
        rois[label_combi] = np.unique(combi)

    return rois


def timepoint_avg_cov(X):
    """Estimate covariance by averaging across time points."""
    T = X.shape[2]
    covs = []
    for t in range(T):
        Xt = X[:, :, t]
        covs.append(LedoitWolf().fit(Xt).covariance_)
    return np.mean(covs, axis=0)


def zscore_baseline(X, bl, eps=1e-8):
    """Standardize each channel using baseline."""
    base = X[:, :, bl[0]:bl[1]]
    mean = base.mean(axis=(0, 2), keepdims=True)
    std = base.std(axis=(0, 2), keepdims=True) + eps
    return (X - mean) / std


def timepoint_pearson_matrix(X, Y):
    """Correlate patterns at every pair of time points."""
    # Rows are time points; columns are spatial features.
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    Xn = Xc / (Xc.std(axis=1, keepdims=True) + 1e-8)
    Yn = Yc / (Yc.std(axis=1, keepdims=True) + 1e-8)
    return Xn @ Yn.T / X.shape[1]


def inv_sqrtm_eigh(Sigma, eps=1e-10):
    """Compute a stable inverse square root for whitening."""
    Sigma = 0.5 * (Sigma + Sigma.T)
    w, v = np.linalg.eigh(Sigma)
    w = np.clip(w, eps, None)
    return (v * (w**-0.5)) @ v.T


def rsa_main():
    """Run condition-wise similarity across tasks and partitions."""

    rois_list = get_rois_list()
    rois = list(rois_list.keys())

    tmin, tmax = -.5, 1.5

    # Preallocate the task x partition x target result tree.
    results = {
        r: {
            t1: {
                t2: {
                    p1: {
                        p2: {
                            target: {} for target in cfg.content_targets
                        } for p2 in cfg.task_partitions[t2]
                    } for p1 in cfg.task_partitions[t1]
                } for t2 in cfg.tasks
            } for t1 in cfg.tasks
        } for r in rois
    }

    # The current analysis uses only the combined six-ROI subset.
    for roi in rois:
        if roi != 'combined_6':
            continue
        rois_subset = {roi: rois_list[roi]}

        # Reconstruct the same ROI subset for both tasks.
        data, mdata, whitening_cache = {}, {}, {}
        for task in cfg.tasks:
            data[task], mdata[task] = get_data_rois_subset(task, rois_subset)

        # Evaluate each unordered task/partition pair once.
        for i, task_a in enumerate(cfg.tasks):
            for j, task_b in enumerate(cfg.tasks):
                if j < i:
                    continue
                for m, part_a in enumerate(cfg.task_partitions[task_a]):
                    for n, part_b in enumerate(cfg.task_partitions[task_b]):
                        if task_a == task_b and n < m:
                            continue

                        # Select the two cognitive partitions to compare.
                        d_task_a, md_task_a = get_partition(data[task_a][roi], mdata[task_a], part_a, task_a)
                        d_task_b, md_task_b = get_partition(data[task_b][roi], mdata[task_b], part_b, task_b)

                        if d_task_a.ndim != 3 or d_task_b.ndim != 3:
                            continue

                        _, n_chans, n_time = d_task_a.shape
                        timepoints = np.linspace(tmin, tmax, n_time)

                        for target in cfg.content_targets:
                            print(f'>>similarity:{roi}: {task_a}-{part_a} vs. {task_b}-{part_b}: {target}')

                            # Split each partition into its two content conditions.
                            c_task_a, conds, count_task_a = get_cdata(d_task_a, md_task_a, target)
                            c_task_b, _, count_task_b = get_cdata(d_task_b, md_task_b, target)

                            n_conds = len(conds)

                            if cfg.whitening:
                                # Estimate and cache one whitening matrix per condition.
                                for task_label, c_data, t_name, p_name in [
                                    ("task_a", c_task_a, task_a, part_a),
                                    ("task_b", c_task_b, task_b, part_b)
                                        ]:
                                    cache_key = (roi, t_name, p_name, target)
                                    if cache_key not in whitening_cache:
                                        sigma = [timepoint_avg_cov(x) for x in c_data]
                                        sigma_inv = [scipy.linalg.fractional_matrix_power(x, -0.5) for x in sigma]
                                        whitening_cache[cache_key] = sigma_inv

                                sigma_a_inv = whitening_cache[(roi, task_a, part_a, target)]
                                sigma_b_inv = whitening_cache[(roi, task_b, part_b, target)]

                                # Apply Sigma^-1/2 to every trial/time spatial pattern.
                                for c in range(n_conds):
                                    c_task_a[c] = np.einsum('ntc,cd->ntd', c_task_a[c].transpose(0, 2, 1), sigma_a_inv[c]).transpose(0, 2, 1)
                                    c_task_b[c] = np.einsum('ntc,cd->ntd', c_task_b[c].transpose(0, 2, 1), sigma_b_inv[c]).transpose(0, 2, 1)
                            else:
                                for c in range(n_conds):
                                    c_task_a[c] = zscore_baseline(c_task_a[c], [0, 25])
                                    c_task_b[c] = zscore_baseline(c_task_b[c], [0, 25])

                            # Correlate condition means at every cross-time pair.
                            sim = np.full((n_conds, n_conds, n_time, n_time), np.nan, dtype=np.float32)
                            for c1 in range(n_conds):
                                mean_a = c_task_a[c1].mean(axis=0)
                                for c2 in range(n_conds):
                                    mean_b = c_task_b[c2].mean(axis=0)
                                    sim[c1, c2] = timepoint_pearson_matrix(mean_a.T, mean_b.T)

                            results[roi][task_a][task_b][part_a][part_b][target] = {
                                'sim': sim,
                                'conds': conds,
                                'n_trials': [count_task_a, count_task_b],
                                'rois': rois_list[roi]
                            }

                            # Fill the reverse comparison by swapping condition and time axes.
                            if task_a != task_b or part_a != part_b:
                                results[roi][task_b][task_a][part_b][part_a][target] = {
                                    'sim': sim.transpose(1, 0, 3, 2),
                                    'conds': conds,
                                    'n_trials': [count_task_b, count_task_a],
                                    'rois': rois_list[roi]
                                }

    out_file = cfg.similarity_file_source_subroi[:-4] + '_core_v10.pkl'
    dump_data(results, out_file)


def rsa_temporal_splits():
    """Repeat the main RSA after temporal splitting in each partition."""

    cfg.n_splits = 2

    rois_list = get_rois_list()
    rois = list(rois_list.keys())

    def make_partitions(task):
        """Append split labels to each cognitive partition."""
        base_parts = cfg.task_partitions[task]
        return [f"{bp}-split{s+1}" for bp in base_parts for s in range(cfg.n_splits)]

    # Expand each cognitive partition into acquisition-time bins.
    task_splits = {t: make_partitions(t) for t in cfg.tasks}

    # Preallocate the split-partition result tree.
    results = {
        r: {
            t1: {
                t2: {
                    p1: {
                        p2: {
                            target: {} for target in cfg.content_targets
                        } for p2 in task_splits[t2]
                    } for p1 in task_splits[t1]
                } for t2 in cfg.tasks
            } for t1 in cfg.tasks
        } for r in rois
    }

    for roi in rois:
        if roi != 'combined_6':
            continue

        rois_subset = {roi: rois_list[roi]}

        # Reconstruct source data once; temporal splits are applied in metadata.
        data = {task: [] for task in cfg.tasks}
        mdata = {task: [] for task in cfg.tasks}
        whitening_cache = {}
        for task in cfg.tasks:
            data[task], mdata[task] = get_data_rois_subset(task, rois_subset)

        # Evaluate each pair of temporal partitions.
        for i, task_a in enumerate(cfg.tasks):
            for j, task_b in enumerate(cfg.tasks):
                if j < i:
                    continue
                for m, part_a in enumerate(task_splits[task_a]):
                    for n, part_b in enumerate(task_splits[task_b]):
                        if task_a == task_b and n < m:
                            continue

                        # Select both cognitive and temporal partitions.
                        d_task_a, md_task_a = get_partition(data[task_a][roi], mdata[task_a], part_a, task_a)
                        d_task_b, md_task_b = get_partition(data[task_b][roi], mdata[task_b], part_b, task_b)

                        if d_task_a.ndim != 3 or d_task_b.ndim != 3:
                            continue

                        _, n_chans, n_time = d_task_a.shape

                        for target in cfg.content_targets:
                            print(f'>>similarity:{roi}: {task_a}/{part_a} vs. {task_b}/{part_b}: {target}')

                            # Split trials by category or hemifield.
                            c_task_a, conds, count_task_a = get_cdata(d_task_a, md_task_a, target)
                            c_task_b, _, count_task_b = get_cdata(d_task_b, md_task_b, target)

                            n_conds = len(conds)

                            if cfg.whitening:
                                # Cache condition-specific whitening matrices for each split.
                                for task_label, c_data, t_name, p_name in [
                                    ("task_a", c_task_a, task_a, part_a),
                                    ("task_b", c_task_b, task_b, part_b)
                                        ]:
                                    cache_key = (roi, t_name, p_name, target)
                                    if cache_key not in whitening_cache:
                                        sigma = [timepoint_avg_cov(x) for x in c_data]
                                        sigma_inv = [scipy.linalg.fractional_matrix_power(x, -0.5) for x in sigma]
                                        whitening_cache[cache_key] = sigma_inv

                                sigma_a_inv = whitening_cache[(roi, task_a, part_a, target)]
                                sigma_b_inv = whitening_cache[(roi, task_b, part_b, target)]

                                # Apply Sigma^-1/2 to every trial/time pattern.
                                for c in range(n_conds):
                                    c_task_a[c] = np.einsum('ntc,cd->ntd', c_task_a[c].transpose(0, 2, 1), sigma_a_inv[c]).transpose(0, 2, 1)
                                    c_task_b[c] = np.einsum('ntc,cd->ntd', c_task_b[c].transpose(0, 2, 1), sigma_b_inv[c]).transpose(0, 2, 1)
                            else:
                                for c in range(n_conds):
                                    c_task_a[c] = zscore_baseline(c_task_a[c], [0, 25])
                                    c_task_b[c] = zscore_baseline(c_task_b[c], [0, 25])

                            # Build condition-by-condition time-generalization matrix.
                            sim = np.full((n_conds, n_conds, n_time, n_time), np.nan, dtype=np.float32)

                            for c1 in range(n_conds):
                                mean_a = c_task_a[c1].mean(axis=0)
                                for c2 in range(n_conds):
                                    mean_b = c_task_b[c2].mean(axis=0)
                                    sim[c1, c2] = timepoint_pearson_matrix(mean_a.T, mean_b.T)

                            results[roi][task_a][task_b][part_a][part_b][target] = {
                                'sim': sim,
                                'conds': conds,
                                'n_trials': [count_task_a, count_task_b],
                                'rois': rois_list[roi]
                            }

                            # Store the reverse comparison without recomputing it.
                            if task_a != task_b or part_a != part_b:
                                results[roi][task_b][task_a][part_b][part_a][target] = {
                                    'sim': sim.transpose(1, 0, 3, 2),
                                    'conds': conds,
                                    'n_trials': [count_task_b, count_task_a],
                                    'rois': rois_list[roi]
                                }

    out_file = cfg.similarity_file_source_subroi[:-4] + f'_temporal_splits_v10.pkl'
    dump_data(results, out_file)


def rsa_core(data, metadata):
    """Compact RSA used by the vertex-selection analyses."""
    from sklearn.covariance import LedoitWolf

    whitening_cache = {}
    _, n_chans, n_time = data['vg']['seen'].shape

    parts = (['vg', 'seen'], ['replay', 'seen-go'], ['replay', 'seen-no-go'])
    n_parts = len(parts)
    n_conds = 2
    n_targets = len(cfg.content_targets)
    # Axes: partition A, partition B, target, condition A, condition B, time A, time B.
    results = np.full((n_parts, n_parts, n_targets, n_conds, n_conds, n_time, n_time), np.nan)
    trial_counts = np.full((n_parts, n_parts, n_targets, n_conds, 2), np.nan)

    # Evaluate the upper triangle of partition pairs.
    for i, part_a in enumerate(parts):
        for j, part_b in enumerate(parts):
            if j < i:
                continue

            # Select the two cognitive partitions to compare.
            d_a, md_a = data[part_a[0]][part_a[1]], metadata[part_a[0]][part_a[1]]
            d_b, md_b = data[part_b[0]][part_b[1]], metadata[part_b[0]][part_b[1]]

            for k, target in enumerate(cfg.content_targets):
                print(f'>>similarity: {part_a} vs. {part_b}: {target}')

                # Split each partition into its two content conditions.
                c_a, conds_a, count_a = get_cdata(d_a, md_a, target)    #c_a: list with len=2, each item is array with shape=(n_trials, n_chans, n_time)
                c_b, conds_b, count_b = get_cdata(d_b, md_b, target)

                if cfg.whitening:
                    # Estimate and cache one whitening matrix per condition.
                    for task_label, c_data, t_name, p_name in [
                        ("task_a", c_a, part_a[0], part_a[1]),
                        ("task_b", c_b, part_b[0], part_b[1])
                            ]:
                        cache_key = (t_name, p_name, target)
                        if cache_key not in whitening_cache:
                            sigma = [timepoint_avg_cov(x) for x in c_data]
                            sigma_inv = [inv_sqrtm_eigh(x) for x in sigma]
                            whitening_cache[cache_key] = sigma_inv

                    sigma_a_inv = whitening_cache[(part_a[0], part_a[1], target)]
                    sigma_b_inv = whitening_cache[(part_b[0], part_b[1], target)]

                    # Apply Sigma^-1/2 to every trial/time spatial pattern.
                    for c in range(n_conds):
                        c_a[c] = np.einsum('ntc,cd->ntd', c_a[c].transpose(0, 2, 1), sigma_a_inv[c]).transpose(0, 2, 1)
                        c_b[c] = np.einsum('ntc,cd->ntd', c_b[c].transpose(0, 2, 1), sigma_b_inv[c]).transpose(0, 2, 1)
                else:
                    for c in range(n_conds):
                        c_a[c] = zscore_baseline(c_a[c], [0, 25])
                        c_b[c] = zscore_baseline(c_b[c], [0, 25])

                # Correlate condition means at every cross-time pair.
                sim = np.full((n_conds, n_conds, n_time, n_time), np.nan, dtype=np.float32)
                for c1 in range(n_conds):
                    mean_a = c_a[c1].mean(axis=0)
                    for c2 in range(n_conds):
                        mean_b = c_b[c2].mean(axis=0)
                        sim[c1, c2] = timepoint_pearson_matrix(mean_a.T, mean_b.T)

                results[i,j,k] = sim
                trial_counts[i,j,k,:,0] = count_a
                trial_counts[i,j,k,:,1] = count_b

    return {
        'similarity': results,
        'trial_counts': trial_counts,
        'parts': parts,
        'targets': cfg.content_targets,
        'n_chans': n_chans,
        'n_time': n_time
    }


if __name__ == '__main__':

    if sys.argv[2] == 'temporal_splits':
        rsa_temporal_splits()
    elif sys.argv[2] == 'main':
        rsa_main()
