import os
import numpy as np

import cfg
from utils import *

def run_cluster_stats(diff_matrix, p_crit=0.05, tail=1):
    """Significant clusters from a one-sample permutation test."""
    import mne
    _, clusters, cluster_pv, _ = mne.stats.permutation_cluster_1samp_test(diff_matrix, tail=tail, out_type='mask')
    sig_clusters = [clusters[x] for x, p in enumerate(cluster_pv) if p < p_crit]
    return sig_clusters if sig_clusters else None

def summarize_diagonal(diff_matrix):
    """Diagonal mean and bootstrap confidence interval."""
    from scipy.stats import bootstrap
    diag = diff_matrix[:, range(diff_matrix.shape[1]), range(diff_matrix.shape[2])]
    boot = bootstrap((diag,), np.mean, vectorized=True, method='basic', axis=0)
    diag_avg = boot.bootstrap_distribution.mean(axis=1)
    diag_err = np.array([boot.confidence_interval.low, boot.confidence_interval.high])
    return diag, diag_avg, diag_err


def similarity_contrast(dat, contrast, diag_only=False, p_crit=0.05):
    """Content similarity against task similarity."""
    out = {}
    for target in cfg.content_targets:
        out[target] = {}

        # Content similarity: same content across different tasks
        content_sims = []
        for c1 in contrast:
            for c2 in contrast:
                if c1 != c2 and c1 in dat[target] and c2 in dat[target][c1]:
                        content_sims.append(dat[target][c1][c2][:, 0, 0])
                        content_sims.append(dat[target][c1][c2][:, 1, 1])

        content_sim = np.nanmean(np.stack(content_sims, axis=0), axis=0)

        # Task similarity: different content within same task
        task_sims = []
        for task in contrast:
            if task in dat[target] and task in dat[target][task]:
                task_sims.append(dat[target][task][task][:, 0, 1])
                task_sims.append(dat[target][task][task][:, 1, 0])

        task_sim = np.nanmean(np.stack(task_sims, axis=0), axis=0)

        # Positive values favour content over task similarity.
        diff = content_sim - task_sim
        out[target]['sim'] = diff

        # Test the full matrix and its temporal diagonal separately.
        sig = run_cluster_stats(diff, p_crit)
        out[target]['sig'] = np.logical_or.reduce(sig) if sig else None

        diag, diag_avg, diag_err = summarize_diagonal(diff)
        out[target]['diag_avg'] = diag_avg
        out[target]['diag_err'] = diag_err

        diag_sig = run_cluster_stats(diag, p_crit)
        if diag_sig:
            sig_mask = np.zeros(diff.shape[1], dtype=bool)
            for c in diag_sig:
                sig_mask[c] = True
            out[target]['diag_sig'] = sig_mask
        else:
            out[target]['diag_sig'] = None

    return out


def content_contrast(dat, contrast, diag_only=False, p_crit=0.05):
    """Content-specific against content-unspecific similarity."""
    out = {}
    for target in cfg.content_targets:
        out[target] = {}

        # Content similarity: same content across different tasks
        content_sims = []
        for c1 in contrast:
            for c2 in contrast:
                if c1 != c2:  # different tasks
                    content_sims.append(dat[target][c1][c2][:, 0, 0])  # content 0 vs 0
                    content_sims.append(dat[target][c1][c2][:, 1, 1])  # content 1 vs 1

        content_sim = np.nanmean(np.stack(content_sims, axis=0), axis=0)

        # Unspecific similarity: different content across different task
        unspec_sims = []
        for c1 in contrast:
            for c2 in contrast:
                if c1 != c2:  # different tasks
                    unspec_sims.append(dat[target][c1][c2][:, 0, 1])  # content 0 vs 1
                    unspec_sims.append(dat[target][c1][c2][:, 1, 0])  # content 1 vs 1

        unspec_sim = np.nanmean(np.stack(unspec_sims, axis=0), axis=0)

        # Positive values favour matched over mismatched content.
        diff = content_sim - unspec_sim
        out[target]['sim'] = diff

        # Test the full matrix and its temporal diagonal separately.
        sig = run_cluster_stats(diff, p_crit)
        out[target]['sig'] = np.logical_or.reduce(sig) if sig else None

        diag, diag_avg, diag_err = summarize_diagonal(diff)
        out[target]['diag_avg'] = diag_avg
        out[target]['diag_err'] = diag_err

        diag_sig = run_cluster_stats(diag, p_crit)
        if diag_sig:
            sig_mask = np.zeros(diff.shape[1], dtype=bool)
            for c in diag_sig:
                sig_mask[c] = True
            out[target]['diag_sig'] = sig_mask
        else:
            out[target]['diag_sig'] = None

    return out


def get_contrast_main(contrast, freq='combined_6'):
    """Aggregate the main subject-level similarity results."""
    out_dict = {}
    tmin, tmax = -.5, 1.5
    tasks = np.array([['vg','seen'], ['replay','seen-go'], ['replay','seen-no-go']])
    tasks_str = ['_'.join(x) for x in tasks]

    # Load the nested subject-level similarity trees.
    results = {}
    for sub in cfg.subjects:
        file = os.path.join(cfg.out_dir, f'controls/similarity_{sub}_core.pkl')
        if os.path.isfile(file):
            results[sub] = load_data(file)[freq]
        else:
            print(f'No file found for: {file}')

    # Stack subjects for every ordered partition pair.
    for target in cfg.content_targets:
        subs = list(results.keys())
        out_dict[target] = {}
        for c1 in tasks:
            key1 = '_'.join(c1)
            out_dict[target][key1] = {}
            for c2 in tasks:
                key2 = '_'.join(c2)
                sims = []
                for s in subs:
                    try:
                        sim = results[s][c1[0]][c2[0]][c1[1]][c2[1]][target]['sim']
                        sims.append(sim)
                    except KeyError:
                        continue
                if sims:
                    out_dict[target][key1][key2] = np.array(sims)

    # Add the shared time axis and requested group contrasts.
    n_time = out_dict['category'][tasks_str[0]][tasks_str[0]].shape[-1]
    out_dict['time'] = np.linspace(tmin, tmax, n_time)
    if contrast == 'similarity':
        out_dict['stats_all'] = similarity_contrast(out_dict, tasks_str)
        out_dict['stats_replay'] = similarity_contrast(out_dict, ['replay_seen-go', 'replay_seen-no-go'])
    elif contrast == 'content':
        out_dict['stats_all'] = content_contrast(out_dict, tasks_str)
        out_dict['stats_replay'] = content_contrast(out_dict, ['replay_seen-go', 'replay_seen-no-go'])

    return out_dict


def get_contrast_timesplit(freq='combined_6', temporal_splits=True, p_crit=.06):
    """Aggregate contrasts from the temporal-split analysis."""
    out_dict = {}
    tmin, tmax = -.5, 1.5

    # Define the two cross-task temporal alignments to compare.
    tasks_split_close = np.array([['vg','seen-split2'], ['replay','seen-go-split1'], ['replay','seen-no-go-split1']])
    tasks_str_close = ['_'.join(x) for x in tasks_split_close]
    tasks_split_far = np.array([['vg','seen-split1'], ['replay','seen-go-split2'], ['replay','seen-no-go-split2']])
    tasks_str_far = ['_'.join(x) for x in tasks_split_far]

    # Load temporal-split results for all available subjects.
    results = {}
    for sub in cfg.subjects:
        file = os.path.join(cfg.out_dir, f'controls/similarity_{sub}_temporal_splits.pkl' if temporal_splits else f'similarity_mnn_rois_subset_{sub}_p1.pkl')
        if os.path.isfile(file):
            results[sub] = load_data(file)[freq]
        else:
            print(f'No file found for: {file}')

    subs = list(results.keys())

    # Stack subject matrices for the close pairing.
    data_close = {}
    for target in cfg.content_targets:
        data_close[target] = {
            '_'.join(c1): {
                '_'.join(c2): np.array([
                    results[s][c1[0]][c2[0]][c1[1]][c2[1]][target]['sim']
                    for s in subs
                ]) for c2 in tasks_split_close
            } for c1 in tasks_split_close
        }

    # Stack the corresponding far pairing.
    data_far = {}
    for target in cfg.content_targets:
        data_far[target] = {
            '_'.join(c1): {
                '_'.join(c2): np.array([
                    results[s][c1[0]][c2[0]][c1[1]][c2[1]][target]['sim']
                    for s in subs
                ]) for c2 in tasks_split_far
            } for c1 in tasks_split_far
        }

    # Compute each contrast before testing their difference.
    n_time = data_close['category'][tasks_str_close[0]][tasks_str_close[0]].shape[-1]
    out_dict['time'] = np.linspace(tmin, tmax, n_time)
    out_dict['stats_close'] = similarity_contrast(data_close, tasks_str_close)
    out_dict['stats_far'] = similarity_contrast(data_far, tasks_str_far)

    # Difference between the two contrasts
    out_dict['stats_diff'] = {}
    for target in cfg.content_targets:
        diff_sim = out_dict['stats_close'][target]['sim'] - out_dict['stats_far'][target]['sim']
        out_dict['stats_diff'][target] = {'sim': diff_sim}
        sig = run_cluster_stats(diff_sim, p_crit)
        out_dict['stats_diff'][target]['sig'] = np.logical_or.reduce(sig) if sig else None
        diag, diag_avg, diag_err = summarize_diagonal(diff_sim)
        out_dict['stats_diff'][target]['diag_avg'] = diag_avg
        out_dict['stats_diff'][target]['diag_err'] = diag_err
        diag_sig = run_cluster_stats(diag, p_crit)
        if diag_sig:
            sig_mask = np.zeros(diff_sim.shape[1], dtype=bool)
            for c in diag_sig:
                sig_mask[c] = True
            out_dict['stats_diff'][target]['diag_sig'] = sig_mask
        else:
            out_dict['stats_diff'][target]['diag_sig'] = None

    return out_dict


def get_contrast_optimal():
    """Aggregate cross-validated results from optimal vertex selection."""
    tmin, tmax = -.5, 1.5
    percentiles = [90, 95]
    n_time = 100

    # Stack subject-level similarity for each selection threshold.
    out_dict = {}
    for percentile in percentiles:
        out_dict[percentile] = {'data': []}
        for s,sub in enumerate(cfg.subjects):         
            file = os.path.join(cfg.out_dir, f'rsa_optimal_p{percentile}_{sub}_AVG.pkl')
            if os.path.isfile(file):
                out_dict[percentile]['data'].append(load_data(file)['similarity'])
            else:
                print(f'No file found for: {file}')
        out_dict[percentile]['data'] = np.array(out_dict[percentile]['data'])

    out_dict['time'] = np.linspace(tmin, tmax, n_time)

    for percentile in percentiles:
        dat = out_dict[percentile]['data']
        # Average each time matrix with its transpose before contrasting.
        dat = np.mean((dat, np.transpose(dat, (0,1,2,3,4,5,7,6))),axis=0) # make symmetric
        out_dict[percentile]['stats_all'] = similarity_contrast(dat, [0,1,2])
        out_dict[percentile]['stats_replay'] = similarity_contrast(dat, [1,2])

        out_dict[percentile]['content_cnt_all'] = content_contrast(dat, [0,1,2])
        out_dict[percentile]['content_cnt_replay'] = content_contrast(dat, [1,2])

    return out_dict


def get_contrast_gamma():
    """Aggregate results from gamma-based vertex selection."""
    tmin, tmax = -.5, 1.5

    n_subs = len(cfg.subjects)
    selections = ['1d', '2d']
    percentiles = [-95, -90, 90, 95]
    n_select = len(selections)
    parts = (['vg', 'seen'], ['replay', 'seen-go'], ['replay', 'seen-no-go'])
    n_parts = len(parts)
    n_targets = len(cfg.content_targets)
    n_conds = 2
    n_time = 100
    # Axes: subject, selection, partition A/B, target, conditions A/B, time A/B.
    results_shape = (n_subs, n_select, n_parts, n_parts, n_targets, n_conds, n_conds, n_time, n_time)

    out_dict = {}
    # Fill group array from available subject files.
    for percentile in percentiles:
        out_dict[percentile] = {'data': np.full(results_shape, np.nan)}
        for s,sub in enumerate(cfg.subjects):       
            for sl,selection in enumerate(selections):     
                file = os.path.join(cfg.out_dir, f'rsa-gamma_multitaper_dif_{selection}_p{percentile}_bin1_subroi_{sub}.pkl')
                if os.path.isfile(file):
                    out_dict[percentile]['data'][s,sl] = load_data(file)['similarity']
                else:
                    print(f'No file found for: {file}')

    out_dict['time'] = np.linspace(tmin, tmax, n_time)

    for percentile in percentiles:
        out_dict[percentile]['stats_all'] = {}
        out_dict[percentile]['stats_replay'] = {}
        dat = out_dict[percentile]['data']
        # Average each time matrix with its transpose before contrasting.
        dat = np.mean((dat, np.transpose(dat, (0,1,2,3,4,5,6,8,7))),axis=0) # make symmetric

        for sl,selection in enumerate(selections):

            out_dict[percentile]['stats_all'][selection] = similarity_contrast(dat[:,sl], [0,1,2])
            out_dict[percentile]['stats_replay'][selection] = similarity_contrast(dat[:,sl], [1,2])

            out_dict[percentile]['content_cnt_all'] = content_contrast(dat[:,sl], [0,1,2])
            out_dict[percentile]['content_cnt_replay'] = content_contrast(dat[:,sl], [1,2])

    return out_dict


def bf01_greater(x, mu=0, r=0.707):
    """Directional BF01 values for positive one-sample effects."""
    from scipy import stats
    from scipy.stats import t, nct, cauchy
    from scipy.integrate import quad

    # Reduce each matrix cell to its one-sample t statistic and sample size.
    tvals = stats.ttest_1samp(x, mu, axis=0, nan_policy="omit").statistic
    ns = np.sum(~np.isnan(x), axis=0)

    out = np.full(tvals.shape, np.nan)

    for idx in np.ndindex(tvals.shape):
        tv, n = tvals[idx], int(ns[idx])
        if np.isnan(tv) or n < 2:
            continue

        df = n - 1
        h0 = t.pdf(tv, df)

        # Integrate the positive-effect alternative over a half-Cauchy prior.
        h1 = quad(lambda d: nct.pdf(tv, df, d * np.sqrt(n)) * 2 * cauchy.pdf(d, scale=r), 0, np.inf)[0]

        out[idx] = h0 / h1

    return out
