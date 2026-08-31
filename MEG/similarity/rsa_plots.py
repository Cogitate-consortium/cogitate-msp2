import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl

import cfg
from rsa_contrast import *

mpl.rcParams.update({"svg.fonttype": 'none'})
plt.rc("axes.spines", top=False, right=False)
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#67c1c1", "#df7712", "#ffd149", "#7e0446"])
cfg.palette = ["#67c1c1","#df7712","#ffd149","#7e0446"]
cfg.hfont = {'fontname': 'Helvetica'}


def plot_similarity(data, fig, contrast):
    """Plot category and location similarity matrices and diagonals."""
    font_sz = 'x-large'
    time = data['time']
    t_dif = np.diff(time)[0] / 2
    plt_extent = tuple([time[0] - t_dif, time[-1] + t_dif] * 2)
    tg_cmap = 'RdBu_r'

    # Average subjects and use one zero-centred scale for both targets.
    mdat = np.array([np.nanmean(data[contrast][t]['sim'], axis=0) for t in cfg.content_targets])
    clim = np.max(np.abs(mdat))
    cnorm = mcolors.TwoSlopeNorm(vmin=-clim, vcenter=0, vmax=clim)
    cticks = np.round([-clim, 0, clim], 1)

    # Top row: time-generalization matrices; bottom row: diagonals.
    axes = fig.subplot_mosaic("""ab
                                cc""")
    for t, target in enumerate(cfg.content_targets):
        ax = axes[list(axes.keys())[t]]
        im = ax.imshow(mdat[t], origin='lower', extent=plt_extent, cmap=tg_cmap, norm=cnorm)
        sig = data[contrast][target].get('sig')
        if sig is not None:
            ax.contour(sig, colors='dimgrey', extent=plt_extent, linewidths=1)

        ax.set_title(target, fontsize=font_sz, **cfg.hfont)
        ax.axhline(0, c='grey', ls='--')
        ax.axvline(0, c='grey', ls='--')
        ax.set_xticks([0, .25, .5, .75, 1, 1.25])
        ax.set_yticks([0, .25, .5, .75, 1, 1.25])
        ax.set_xticklabels(['0','.25','.5','.75','1','1.25'], fontsize=font_sz, **cfg.hfont)
        ax.set_yticklabels(['0','.25','.5','.75','1','1.25'], fontsize=font_sz, **cfg.hfont)
        ax.set_xlabel('time (s)', fontsize=font_sz, **cfg.hfont)
        if t == 0:
            ax.set_ylabel('time (s)', fontsize=font_sz, **cfg.hfont)

        # Diagonal plot
        ax_diag = axes['c']
        ax_diag.plot(time, data[contrast][target]['diag_avg'], label=target, color=cfg.palette[t])
        ax_diag.fill_between(time,
                             data[contrast][target]['diag_err'][0],
                             data[contrast][target]['diag_err'][1],
                             alpha=0.3, color=cfg.palette[t])
        ax_diag.set_ylim(-.4, .3)

        diag_sig = data[contrast][target].get('diag_sig')
        if diag_sig is not None:
            sig_time = np.where(diag_sig, time, np.nan)
            ax_diag.scatter(sig_time, [(-clim + 0.05 - t*0.02)] * len(time), marker='.', color=cfg.palette[t], s=50)

    clb = fig.colorbar(im, ax=axes['b'], orientation='vertical', ticks=cticks)
    clb.set_label('similarity (pearson r)', size=font_sz)
    clb.set_ticklabels([f'{x:.1f}'.replace('.0','0') for x in cticks])

    ax_diag.set_xlabel('time (s)', fontsize=font_sz, **cfg.hfont)
    ax_diag.set_ylabel('similarity (pearson r)', fontsize=font_sz, **cfg.hfont)
    ax_diag.axhline(0, color='grey', linestyle='-', alpha=0.5)
    ax_diag.axvline(0, color='grey', linestyle='-', alpha=0.5)
    ax_diag.legend(loc='upper right', frameon=False, prop={"family": "Helvetica", "size": font_sz})


def plot_comparison(dat, bl, opt, times, labels, title):
    """Compare diagonal time courses across vertex-selection strategies."""

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5,8),sharex=True, sharey=True)

    axs[0].axhline(0, linestyle='-', c='grey')
    axs[0].axvline(0, linestyle='-', c='grey')
    axs[1].axhline(0, linestyle='-', c='grey')
    axs[1].axvline(0, linestyle='-', c='grey')
    axs[0].set_title('category')
    axs[1].set_title('location')
    
    # Main analysis serves as the reference trajectory.
    axs[0].plot(times, bl['category']['diag_avg'][::2], label='main analysis', linestyle='--')
    axs[0].fill_between(times, bl['category']['diag_err'][0][::2], bl['category']['diag_err'][1][::2], alpha=.3)
    axs[1].plot(times, bl['location']['diag_avg'][::2], label='main analysis', linestyle='--')
    axs[1].fill_between(times, bl['location']['diag_err'][0][::2], bl['location']['diag_err'][1][::2], alpha=.3)

    # Add gamma-selected and random-control trajectories.
    for k,key in enumerate(list(dat.keys())):
        axs[0].plot(times, dat[key]['category']['diag_avg'], label=labels[k])
        axs[0].fill_between(times, dat[key]['category']['diag_err'][0], dat[key]['category']['diag_err'][1], alpha=.3)

        axs[1].plot(times, dat[key]['location']['diag_avg'], label=labels[k])
        axs[1].fill_between(times, dat[key]['location']['diag_err'][0], dat[key]['location']['diag_err'][1], alpha=.3)

    # Add the cross-validated optimal-vertex result.
    for key in [95]:
        x = opt[key]['stats_all']
        axs[0].plot(times, x['category']['diag_avg'], label=f'optimised top 5%')
        axs[0].fill_between(times, x['category']['diag_err'][0], x['category']['diag_err'][1], alpha=.3)
        axs[1].plot(times, x['location']['diag_avg'], label=f'optimised top 5%')
        axs[1].fill_between(times, x['location']['diag_err'][0], x['location']['diag_err'][1], alpha=.3)

    axs[0].set_xlim([-0.5, 1.5])
    axs[1].set_xlim([-0.5, 1.5])

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=2, frameon=False, bbox_transform=fig.transFigure)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.07, 1, 1.02])
    plt.show()


def plot_bf01(bf, t):
    """Plot diagonal and full-matrix Bayes factors."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    # Diagonal BF01 over time.
    ax[1].plot(t, np.diagonal(bf))
    ax[1].axhline(1, ls='--', lw=1, color='k', label='BF01 = 1')
    ax[1].axhline(5, ls=':', lw=1, color='k', label='BF01 = 5')
    ax[1].set_title('diagonal')
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('BF01: H0 / H1')
    ax[1].legend(frameon=False, loc='upper left')

    # Full BF01 matrix.
    im = ax[0].imshow(bf, origin='lower', aspect='auto', extent=[t[0], t[-1], t[0], t[-1]])
    ax[0].set_title('category')
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('time')

    cbar = fig.colorbar(im, ax=ax[0])

    fig.suptitle('Bayes factors for category similarity', y=1.05)
    plt.show()


def main():
    """Load group contrasts and generate the analysis figures."""

    # Main all-task and replay-only contrasts.
    data = get_contrast_main('similarity')
    for stat_key in ['stats_all', 'stats_replay']:
        fig = plt.figure(layout='compressed', figsize=(10, 8))
        plot_similarity(data, fig, stat_key)
        plt.show()
 
    # Temporal-proximity control.
    data_ts = get_contrast_timesplit(temporal_splits=True)
    for stat_key in ['stats_close', 'stats_far']:
        if stat_key not in data_ts:
            continue
        fig = plt.figure(layout='compressed', figsize=(10, 8))
        plot_similarity(data_ts, fig, stat_key)
        plt.show()

    # Gamma-selected vertex analysis.
    data_gamma = get_contrast_gamma()
    time_labs = data_gamma['time']
    percentiles_str = ['random 5%','random 10%','gamma top 10%','gamma top 5%']
    percentiles_og = [-95,-90,90,95]
    keymap = {percentiles_og[x]: percentiles_str[x] for x in range(len(percentiles_og))}
    keymap['time'] = 'time'
    data_gamma = {keymap[k]: v for k,v in data_gamma.items()}
    for percentile in ['gamma top 5%']:
        for selection in ['1d']:
            for stat_key in ['stats_all']:
                dat = data_gamma[percentile][stat_key][selection]
                fig = plt.figure(layout='compressed', figsize=(4, 3.2))
                plot_data = {'time': time_labs, 'stats_all': dat}
                plot_similarity(plot_data, fig, 'stats_all')
                plt.show()

    # Cross-validated optimal-vertex analysis.
    opt_data = get_contrast_optimal()
    fig = plt.figure(layout='compressed', figsize=(4, 3.2))
    plot_data = {'time': time_labs, 'stats_all': opt_data[95]['stats_all']}
    plot_similarity(plot_data, fig, 'stats_all')
    plt.show()
    
    # Compare main, gamma-selected, random, and optimal diagonals.
    dat = {}
    selection = '1d'
    stat_key = 'stats_all'
    title = ''
    bl = data[stat_key]
    labs_str = ['random 5%','gamma top 5%']
    for var in labs_str:
        dat[var] = data_gamma[var][stat_key][selection]   
    plot_comparison(dat, bl, opt_data, time_labs, labs_str, title)

    # Bayes-factor map for the category contrast.
    bf = bf01_greater(data['stats_all']['category']['sim'])
    t = data['time']
    plot_bf01(bf, t)



