import pandas as pd
import scipy
from scipy.stats import norm
import pingouin as pg
import math
import random
import numpy as np
import sys
import lineplotter

"""
This module is in charge of performing all statistical tests on given datasets. It is agnostic to the data's meaning, 
it just performs tests and assumption checks on given columns which are given as input. It heavily relies on Pingouin
package. Currently this module includes:
- paired t test: paired_t_test_pg
- independent t test: independent_t_test_pg
- one way ANOVA: one_way_anova
- repeated measures ANOVA: repeated_measures_anova
- cluster-mass permutation t-test: permutation_cluster_paired_ttest: see details in method documentation
- correlation (Pearson, Spearman, Kendall, ...): corr_test
- Chi squared test: chi_squared
deprecated:
- paired_t_test [deprecated, use paired_t_test_pg as it includes Bayes Factors and outputs the same results]: 
including a preliminary Shapiro-Wilk test to check the normal distribution assumption
- wilcoxon_test: to compare paired-samples which violate the normal distribution assumption

@author: RonyHirsch
"""

p_val = 0.05
WILCOXON = 'W'
TTEST = 't'
ANOVA = "ANOVA"
ANOVA_STAT = "F"


def summary_message(p):
    if p < p_val:
        print(f"p < {p_val}, statistically significant")
    else:
        print(f"p >= {p_val}, not statistically significant")


def wilcoxon_test(data: pd.DataFrame, col_1, col_2):
    result = [WILCOXON]
    # in the (super unrealistic) case where the game column is identical to the replay column, no reason to test
    # in scipy, the result is a tuple in which the first value is the statistic and the second is the p-value
    try:
        W_res = scipy.stats.wilcoxon(data[col_1], data[col_2])
        print(f"WILCOXON SIGNED RANK TEST ON {col_1}, {col_2}: RESULT: W={W_res[0]}, p={W_res[1]}")
        summary_message(W_res[1])
        result.extend([W_res[0], W_res[1]])
    except ValueError:
        print(f"COLUMNS ARE IDENTICAL: cannot use scipy's wilcoxon to test for significance")
        result.extend([None, None])
    return result


def paired_t_test(data: pd.DataFrame, col_1, col_2, stop_if_violated=True):
    result = [TTEST]
    # STEP 1: use Shapiro-Wilk test to test that the data is approximately normally distributed
    # in scipy, the result is a tuple in which the first value is the statistic and the second is the p-value
    try:
        sw_col1 = scipy.stats.shapiro(data[col_1])
        sw_col2 = scipy.stats.shapiro(data[col_2])
    except ValueError:
        print(f"PAIRED T-TEST ON {col_1}, {col_2}: "
              f"Not enough data samples to perform Shapiro-Wilk test")
        print("Using Wilcoxon signed rank instead.")
        result = wilcoxon_test(data, col_1, col_2)
        return result
    if sw_col1[1] < p_val and sw_col2[1] < p_val:
        print(f"PAIRED T-TEST ON {col_1}, {col_2}: "
              f"Violation of assumption of normality (Shapiro-Wilk p vals:{sw_col1[1]}, {sw_col2[1]})")
        if stop_if_violated:
            print("Using Wilcoxon signed rank instead.")
            result = wilcoxon_test(data, col_1, col_2)
            return result
        else:
            print("Continuing anyway")
    # STEP 2: PAIRED SAMPLES T-TEST : this is a TWO SIDED test for the null hypothesis.
    # in scipy, the result is a tuple in which the first value is the W and the second is the p-value
    t_res = scipy.stats.ttest_rel(data[col_1], data[col_2])
    print(f"PAIRED T-TEST ON {col_1}, {col_2}: RESULT: t={t_res[0]}, p={t_res[1]}")
    summary_message(t_res[1])
    result.extend([t_res[0], t_res[1]])
    return result


def paired_t_test_pg(data: pd.DataFrame, col_1, col_2, df=False):
    # NOTE: with pingouin package, the default Cauchy scale factor for computing the Bayes Factor is 0.707
    # to change that, insert "r=..." in the ttest function
    stat = None
    if data.shape[0] >= 2:
        res = pg.ttest(x=data[col_1], y=data[col_2], paired=True)
        res.columns = res.columns.str.replace('%', '')
        if not df:
            stat = res.iloc[0]
        else:
            stat = res
    else:
        print(f"Not enough data samples to perform t-test with Bayes Factors")
    return stat


def independent_t_test_pg(data: pd.DataFrame, col_1, col_2):
    # NOTE: with pingouin package, the default Cauchy scale factor for computing the Bayes Factor is 0.707
    # to change that, insert "r=..." in the ttest function
    stat = None
    if data.shape[0] >= 2:
        res = pg.ttest(x=data[col_1], y=data[col_2], paired=False)
        res.columns = res.columns.str.replace('%', '')
        stat = res
    else:
        print(f"Not enough data samples to perform t-test with Bayes Factors")
    return stat


def correct_multi_comp(pvals, alpha=0.05, method="bonf"):
    # https://pingouin-stats.org/generated/pingouin.multicomp.html
    reject, pvals = pg.multicomp(pvals=pvals, alpha=alpha, method=method)
    return reject, pvals


def one_way_anova(data_name, data: pd.DataFrame, columns):
    result = None
    if data.shape[0] >= 2:
        group_col = list()
        data_col = list()
        for col in columns:
            group_data = list(data[col])
            group_name = [col] * len(group_data)
            group_col.extend(group_name)
            data_col.extend(group_data)
        total_data = pd.DataFrame({'data': data_col, 'group': group_col})
        # dv = name of the column containing the dependent variable
        # between = name of column containing the between-factor
        stat = pg.anova(data=total_data, dv='data', between='group', detailed=True)
        stat = stat.rename(columns={"SS": "sum_squares", "DF": "dof", "MS": "mean_squares", "p-unc": "pval", "np2": "partial_eta_sq"})
        result = stat[stat.Source != 'Within']
        print(f"ONE-WAY ANOVA ON {data_name} {columns}: RESULT: F={list(result['F'])[0]}, p={list(result['pval'])[0]}")
    return result


def repeated_measures_anova(data_name, data: pd.DataFrame, columns, id_col="Subject"):
    cols = [id_col] + columns
    subject_data = data[cols]
    tukey = None  # only if our data is significant, we'll do the Tukey post-hoc test
    # reshape the data for analysis
    subject_data_melt = pd.melt(subject_data.reset_index(), id_vars=[id_col], value_vars=columns)
    # replace column names
    subject_data_melt.columns = [id_col, "Condition", "Data"]
    # dv = dependent variable
    stat = pg.rm_anova(data=subject_data_melt, dv="Data", within="Condition", subject=id_col, detailed=True, correction=True)
    result = stat.rename(
        columns={"SS": "sum_squares", "DF": "dof", "MS": "mean_squares", "p-unc": "pval",
                 "p-GG-corr": "GG_correction_pval", "eps": "GG_correction_epsilon_factor_sphericity_index",
                 "np2": "partial_eta_sq",
                 "W-spher": "sphericity_test", "p-spher": "pval_of_sphericity_test"})
    print(f"ONE-WAY REPEATED MEASURES ANOVA ON {data_name} {columns}: RESULT: F={list(result['F'])[0]}, "
          f"p={list(result['pval'])[0]} ")
    # repeated measures ANOVA assumptions:
    # sphercity: The violation of the assumption of sphericity can lead to an increase in type II error
    # (loss of statistical power) and the F value is not valid.
    # see: https://www.reneshbedre.com/blog/repeated-measure-anova.html
    verdict = "VIOLATED" if result.loc[0, "pval_of_sphericity_test"] > 0.05 else "PASSED"
    print(f"SPHERCITY TEST: {verdict}")

    # if the ANOVA is significant, the Tukey-HSD post-hoc test can let us know who's fault is that
    if result.loc[0, "pval"] < 0.05:
        tukey = pg.pairwise_tukey(data=subject_data_melt, dv="Data", between="Condition", effsize='AUC')

    return result, tukey


def scipy_paired_ttest_only_tval(a, b, axis=0, nan_policy="omit", alternative="two-sided"):
    """
    This method is used as an input method for "paired_permutation_cluster_test" statistical test (stat_fun parameter).
    The reason we don't just put "stat_fun=scipy.stats.ttest_rel" in "mne.stats.permutation_cluster_test", is that the
    expected output of stat_fun according to the documentation is JUST the statistic, where scipy.stats.ttest_rel
    outputs both the statistic and the p-value.
    - mne.stats.permutation_cluster_test: https://mne.tools/stable/generated/mne.stats.permutation_cluster_test.html
    - the default stat_fun documentation: https://mne.tools/stable/generated/mne.stats.f_oneway.html#mne.stats.f_oneway
    - scipy's ttest_rel:
    https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.stats.ttest_rel.html#scipy.stats.ttest_rel
    :return: the statistic value of the result of a a two-sided test for the null hypothesis that 2 related or repeated
    samples have identical average (expected) values.
    """
    stat, pval = scipy.stats.ttest_rel(a=a, b=b, axis=axis, nan_policy=nan_policy, alternative=alternative)
    return stat


def progress(count, total, status='Progress'):
    # credit to @vladignatyev, taken from : https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.1 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write(f"\r[{bar}] {percents}% ...{status}")
    sys.stdout.flush()
    return


def permutation_cluster_paired_ttest(data_cond_1, data_cond_2, p_threshold=0.05, permutations=1024, tail=0,
                                     plot_path=None, plot_name=None):
    """
    Perform a cluster permutation test PAIRED t-test, based on this paper:
    Eric Maris and Robert Oostenveld. Nonparametric statistical testing of EEG- and MEG-data.
    Journal of Neuroscience Methods, 164(1):177–190, 2007. doi:10.1016/j.jneumeth.2007.03.024.
    NOTE : this method assumes data_cond_1 and data_cond_2 contain the SAME COLUMNS, each column represents a subject
    and the same rows (each row represents a timepoint), with the only thing changing between 1 and 2 is the values.

    The way this function works:
    based on : https://benediktehinger.de/blog/science/statistics-cluster-permutation-test/, adjusted for TWO SIDED
    t-test and for acommodation of more than 1 cluster
    - Step 1: calculate the difference between conditions per time point
    - Step 2: test statistics  (t-test)
    - Step 3: clusters over time: define clusters by an arbitrary threshold and test whether these clusters are larger
    clusters that occur by chance. We do not want to do a statistical test for each time-point individually, because we
    would need to correct for multiple comparison for all timepoints. Instead, we define clusters by an arbitrary
    threshold (based on p_threshold) and test whether these clusters are larger clusters that occur by chance.
    - Step 4: calculate cluster-level statistics by taking the SUM of the t_values within each cluster
    (Maris & Oostenveld). The cluster mass is the largest of the cluster-level statistics (if there are multiple
    clusters).
    - Step 5: permutation of the data: shuffle the condition-label for each subject. Note that we actually do not need
    to go back to the two conditions, but we could just flip (multiply by -1) randomly every subject-difference curve in
    diff_df, and calculate the cluster size.

    How is the t-threshold calculated:
    The P-value (p_threshold) would be the probability under the density curve of Student's t distribution with n−1
    degrees of freedom beyond the observed t statistic. The value t we wish to reclaim from the reported p is then the
    inverse CDF (quantile) function of 1−p.
    # ppf: Percent Point Function (Inverse of CDF)
    # https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html

    :param data_cond_1: dataframe where columns are subjects and rows are timepoints, in the first condition
    :param data_cond_2: dataframe where columns are subjects and rows are timepoints, in the second condition
    :param permutations: number of permutations to perform
    :param tail: 0 two tail, 1 one tail - right now not in use as I assume TWO TAIL
    :return:
    """

    # Calculate the difference between conditions per time point
    diff_df = pd.DataFrame()  # create a df that is the difference between data_cond 1 and 2 (= 2 - 1)
    for i in range(data_cond_1.shape[1]):  # iterate over columns = subjects (rows=timepoints)
        sub_name = data_cond_1.columns[i]
        sub_cond_1 = data_cond_1.iloc[:, i]
        sub_cond_2 = data_cond_2.iloc[:, i]
        sub_diff = sub_cond_2 - sub_cond_1
        diff_df[sub_name] = sub_diff

    # Calculate the appropriate t-value threshold:
    deg_of_freedom = data_cond_1.shape[1]  # amount of subjects = amount of columns in original dataframe
    actual_p = p_threshold / 2  # TWO-SIDED HYPOTHESIS!!!!
    t_threshold_pos = abs(scipy.stats.t.ppf([actual_p], [deg_of_freedom])[0])
    t_threshold_neg = -t_threshold_pos

    # first, let's calculate the observed cluster/s that we have in the data. We will then test our permutation clusters
    # against EACH observed cluster
    ttest_sums, largest_cluster_ttest, cluster_full_df = ttest_clusters_over_time(diff_df=diff_df,
                                                                                    t_threshold_pos=t_threshold_pos,
                                                                                    t_threshold_neg=t_threshold_neg)
    # Permutations
    permutation_clusters_list = list()
    cluster_mass_list = list()
    for i in range(permutations):
        progress(count=i, total=permutations, status='Permutation cluster mass t-test progress')
        # create dataframe of diffs
        # Shuffle the condition-label for each SUBJECT (cond1 turns to cond 2 and vice versa) by flipping
        # (multiplying by -1) randomly every subject-difference curve
        perm_diff_df = diff_df.copy()
        for col in perm_diff_df.columns:
            perm_diff_df[col] = perm_diff_df[col] * random.choice([-1, 1])
        #perm_diff_df = pd.DataFrame([each*random.choice([-1, 1]) for each in diff_df.to_numpy()])
        #perm_diff_df.columns = diff_df.columns
        sum_df, cluster_mass, perm_cluster_full_df = ttest_clusters_over_time(diff_df=perm_diff_df,
                                                                              t_threshold_pos=t_threshold_pos,
                                                                              t_threshold_neg=t_threshold_neg)
        del perm_cluster_full_df  # we don't use it here
        # add to lists
        permutation_clusters_list.append(sum_df)
        cluster_mass_list.append(cluster_mass)

    # We now check whether our observed cluster mass (first run) is greater than p_threshold% of what we would expect
    # by chance (permutations)
    clusters = list()
    clusters_mass_size = list()
    cluster_p_vals = list()
    clusters_starts = list()
    clusters_ends = list()
    sig_list = list()
    for ind, row in ttest_sums.iterrows():
        # for each cluster we originally observed, check if its cluster mass is greater than the expected p_threshold%
        if ind != 0:  # 0 is NOT a cluster
            check_tail = [1 if row['tval'] > chance_mass else 0 for chance_mass in cluster_mass_list]
            # The exact value gives us the p-value of the cluster, the probability that cluster-mass with the observed size
            # would have occured when there was no actually difference between the conditions.
            probability = sum(check_tail) / len(check_tail)
            p = 1 - probability
            # div p-threshold because this is a TWO-TAIL hypothesis
            significance = f" < {p_threshold / 2} SIGNIFICANT" if p < (p_threshold / 2) else f" > {p_threshold / 2} NOT SIGNIFICANT"
            print(f"\nObserved cluster {ind}: mass={row['tval']:.5f} , cluster p-val={p}: {significance}")
            clusters.append(ind)
            cluster_p_vals.append(p)
            clusters_mass_size.append(row['tval'])
            clusters_starts.append(row['cluster_start'])
            clusters_ends.append(row['cluster_end'])
            sig_list.append(p < (p_threshold / 2))
    clusters_pvals = pd.DataFrame({"cluster": clusters, "mass_size": clusters_mass_size, "pval": cluster_p_vals,
                                   "significant": sig_list, "cluster_starts": clusters_starts, "cluster_ends": clusters_ends})
    cluster_mass_df = pd.DataFrame({"cluster_mass_in_permutation": cluster_mass_list})

    if plot_path is not None:
        lineplotter.plot_avg_line(title="Cluster Permutation Paired T-Test", trial_df_list=[cluster_mass_df],
                                  avg_col_list=["tval"], se_col_list=[None], label_list=[""], y_name=f"t-value",
                                  color_list=["tab:blue"], save=True, save_path=plot_path, save_name=plot_name,
                                  sub_folder="")

    return clusters_pvals, cluster_full_df, cluster_mass_df


def ttest_clusters_over_time(diff_df, t_threshold_pos, t_threshold_neg):
    # calculate the test statistics per timepoint
    diff_df["mean"] = diff_df.mean(axis=1)  # average per row (timepoint) across columns (subject)
    diff_df["std"] = diff_df.std(axis=1)  # std per row (timepoint) across columns (subject)
    diff_df["tval"] = diff_df.apply(t_stat, axis=1)  # our test statistics

    # clusters over time. Cluster-mass statistics : the SUM of t-values
    diff_df["is_above_threshold"] = diff_df.apply(lambda row: thresh_check(row["tval"], t_threshold_pos,
                                                                                     t_threshold_neg), axis=1)
    perm_diff_df = clusters(diff_df)  # separate clusters
    # in sum_df, row per cluster (0 is NOT a cluster), tval is the sum of tvals in this cluster
    # (observed cluster mass)

    perm_diff_df['ind'] = perm_diff_df.index
    cluster_start = perm_diff_df.groupby(['cluster']).ind.idxmin()
    cluster_end = perm_diff_df.groupby(['cluster']).ind.idxmax()
    sum_df = perm_diff_df.groupby(['cluster']).apply(lambda c: c.abs().sum())  # TWO-SIDED HYPOTHESIS: ABSOLUTE SUM
    # If sum_df["is_above_threshold"] == 0 THEN THIS IS -NOT- A CLUSTER! NOTHING WAS ABOVE THRESHOLD
    sum_df.loc[sum_df.is_above_threshold == 0, ['tval']] = 0
    sum_df = sum_df["tval"].to_frame()  # only summation of tvals of the clusters is interesting
    for index, value in cluster_start.items():
        sum_df.loc[index, 'cluster_start'] = value
    for index, value in cluster_end.items():
        sum_df.loc[index, 'cluster_end'] = value
    # the cluster mass is the largest of the cluster-level statistics (if there are multiple clusters)
    cluster_mass = max(sum_df["tval"])
    return sum_df, cluster_mass, perm_diff_df


def t_stat(row):
    excess_columns = 2  # "mean" and "std" columns
    n = len(row) - excess_columns  # all subject columns
    if row["std"] == 0 or math.sqrt(n) == 0:
        t = 0
    else:
        t = row["mean"] / (row["std"] / math.sqrt(n))
    return t


def thresh_check(tval, thresh_pos=None, thresh_neg=None):
    # return 1 if tval is above threshold, 0 otherwise
    result = 0
    if thresh_pos is None:
        result = 1 if tval < thresh_neg else 0
    if thresh_neg is None:
        result = 1 if tval > thresh_pos else 0
    elif thresh_pos is not None and thresh_neg is not None:  # both thresh_pos and thresh_neg
        result = 1 if (tval > thresh_pos or tval < thresh_neg) else 0
    return result


def clusters(cluster_dataframe, cluster_col="is_above_threshold", tval_col="tval"):
    cluster_list = list()
    in_cluster = False  # boolean, are we right now in a cluster or not
    cluster_number = 0  # the index of the current (or last) cluster
    # from Maris & Oostenveld: "for a two-sided test, the clustering is performed separately for samples with a
    # positive and a negative t-value."
    cluster_sign = None  # the SIGN of the cluster

    for ind, row in cluster_dataframe.iterrows():
        if row[cluster_col] == 1:  # we are creating/in a cluster
            if not in_cluster:  # there wasn't an active cluster
                in_cluster = True
                cluster_number += 1
                cluster_sign = np.sign(row[tval_col])  # either 1 (pos), 0 (0), or -1 (neg)
            if in_cluster:  # there was an active cluster
                if cluster_sign is None:  # first time we have a cluster in the data
                    cluster_sign = np.sign(row[tval_col])
                elif cluster_sign != np.sign(row[tval_col]):  # if current cluster sign is NOT identical to our sign
                    in_cluster = True
                    cluster_number += 1  # we are in a SEPARATE cluster, with our sign now
                    cluster_sign = np.sign(row[tval_col])
            # but anyway:
            curr = cluster_number
        else:
            curr = 0  # not a cluster at all
            in_cluster = False
        cluster_list.append(curr)
    cluster_dataframe["cluster"] = cluster_list
    return cluster_dataframe


def corr_test(data_name, data, col1, col2, corr_method):
    # possible corr_methods:
    # 'pearson'
    # 'spearman', 'kendall'
    # 'bicor', 'percbend', 'shepherd', 'skipped'
    try:
        stat = pg.corr(x=data[col1], y=data[col2], alternative='two-sided', method=corr_method)
    except AssertionError:
        print(f"{corr_method} CORRELATION ON {data_name} FAILED: data needs to contain more than 1 element")
        return
    stat = stat.rename(
        columns={"n": "N", "r": "Correlation_Coeff_r", "CI95%": "CI95", "r2": "r_squared", "adg_r2": "Adjusted_r_squared"})
    print(f"{corr_method} CORRELATION ON {data_name} : FULL MATRIX")
    return stat


def chi_squared(data_name, data, col1, col2):
    expected, observed, stat = pg.chi2_independence(data=data, x=col1, y=col2)
    return expected, observed, stat


def SDT(hits, misses, fas, crs):
    """ returns a dict with d-prime measures given hits, misses, false alarms, and correct rejections"""
    Z = norm.ppf
    if fas + crs == 0 or hits + misses == 0: # missing data
        out = dict()
        out['d'] = [np.nan]
        out['beta'] = [np.nan]
        out['c'] = [np.nan]
        out['Ad'] = [np.nan]
        result = pd.DataFrame(out)

    else:
        # Floors an ceilings are replaced by half hits and half FA's
        half_hit = 0.5 / (hits + misses)
        half_fa = 0.5 / (fas + crs)

        # Calculate hit_rate and avoid d' infinity
        hit_rate = hits / (hits + misses)
        if hit_rate == 1:
            hit_rate = 1 - half_hit
        if hit_rate == 0:
            hit_rate = half_hit

        # Calculate false alarm rate and avoid d' infinity
        fa_rate = fas / (fas + crs)
        if fa_rate == 1:
            fa_rate = 1 - half_fa
        if fa_rate == 0:
            fa_rate = half_fa

        # Return d', beta, c and Ad'
        out = dict()
        out['d'] = [Z(hit_rate) - Z(fa_rate)]
        out['beta'] = [math.exp((Z(fa_rate) ** 2 - Z(hit_rate) ** 2) / 2)]
        out['c'] = [-(Z(hit_rate) + Z(fa_rate)) / 2]
        out['Ad'] = [norm.cdf(out['d'][0] / math.sqrt(2))]
        result = pd.DataFrame(out)
    return result


def FDR(stat_name_list, p_val_list, alpha=0.05, method='fdr_bh', save_path=None):
    """
    Correct p-values for multiple comparisons
    :param p_val_list: list of p-vals to correct
    :param alpha: significance level
    :param method: from the methods in https://pingouin-stats.org/generated/pingouin.multicomp.html ,
    default is Benjamini-Hochberg FDR correction
    :return:
    """
    reject_H0, corrected_p_vals = pg.multicomp(p_val_list, alpha=alpha, method=method)
    result = pd.DataFrame({"effect name": stat_name_list, "p-value": p_val_list,
                           "corrected p-value": corrected_p_vals, "reject H0": reject_H0})

    if save_path:
        import os
        result.to_csv(os.path.join(save_path, "FDR_correction.csv"))
    return result
