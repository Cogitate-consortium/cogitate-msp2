import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from functools import reduce
import os
import stats
import quality_checks
import boxplotter
import data_saver
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

"""
This module manages all the analyses related to the relationship between visibility of stimuli in the video game, 
and the visual features of the video game itself. Note that the visual features data is derived from the AnalyzerOutput
results, as the subjects' "Details" files do not include that information in a trial-based manner and in such detail.

@author: RonyHirsch
"""

WORLD = 'world'
LOC_WORLD = 'L'
PROBED = 'isProbed'
response_eval = 'responseEvaluation'
FN = 'FalseNegative'
TN = 'TrueNegative'
TP = 'TruePositive'
FP = 'FalsePositive'
RATE = "Rate"
BF_TTEST_PAIRED = "BF_ttest_paired"
IRREL_ADDITION = 'AdditionalPress'

# kernels
LINEAR = 'linear'
RBF = 'RBF'
POLY = 'polynomial'
SIG = 'sigmoid'


def unify_subs_for_analysis(sub_dict, filter_out_replay=False):
    all_subs_stimanalysis = list()
    for sub in sub_dict.keys():
        sub_analyzer = sub_dict[sub].full.analyzer_output
        if sub_analyzer is not None:
            if not sub_analyzer.empty:
                if filter_out_replay:
                    # no practice levels, AND no replay (localizer) levels - we are interested in the game only
                    sub_analyzer = sub_analyzer[(sub_analyzer[WORLD] != 0) & (sub_analyzer[WORLD] != LOC_WORLD)]
                all_subs_stimanalysis.append(sub_analyzer)
    stim_analysis = pd.concat(all_subs_stimanalysis)
    return stim_analysis


def analyze_vg_effects_on_vis(sub_dict, save_path="", drop_columns=None, test_size=0.3, kernel='linear', name="all",
                              n_permutations=1000, normalize=False, scoring_method="balanced_accuracy", alt=False):
    """

    :param sub_dict:
    :param save_path:
    :param drop_columns:
    :param test_size:
    :param kernel:
    - linear (both the default, and a special case where I don't call "regular" SVC with a linear kernel
    because it takes forever, rather call LinearSVC instance which uses a different optimization and is much faster)
    - kernels that work as an input to the "kernel" parameter of the sklearn's SVC: 'rbf', 'poly', 'sigmoid'
    :return:
    """

    print("SVM on Video Game Features")

    # path to save results
    folder_path = data_saver.create_analysis(save_path)
    if alt:
        features_folder = os.path.join(folder_path, "vg_features_alt")
    else:
        features_folder = os.path.join(folder_path, "vg_features")

    # start with getting all the data into one place
    stim_analysis = unify_subs_for_analysis(sub_dict, filter_out_replay=True)

    # STEP 0: PRE-PROCESS
    # we are only interested in stimuli which were probed
    stim_analysis_probed = stim_analysis[stim_analysis[PROBED] == True]
    # remove all columns that will not be used as features or y:
    """
    Either they are meaningless (e.g., 'versionString')
    OR they are related to y (e.g., 'responseTS')
    OR they are redundant because there are other columns with the same information 
    (e.g., 'avgNumFallingEssenses_m1000_p500')
    OR they reflect raw (unprocessed) eye-tracking data from eyelink which we do not want to consider
    (e.g., 'numBlinks_m2000_p500')
    OR they are related to non-game stuff (e.g., subjectID, world, level, stimType, stimName)
    """
    interesting_cols = ["numFallingEssensesTopHalf_0", "numFallingEssensesBotHalf_0", "avgNumSameColorEssenses_0_p500",
                        "avgNumOppositeColorEssenses_0_p500",
                        "numButtonPresses_0_p500", "spatialProxFromStimLocToPlayer_0", "tempProxOfLastCollision",
                        "tempProxOfLastAbsorption", "spatialProxFromStimLocToNearestEssense_0",
                        "spatialProxFromStimLocToClusterMeanEssenses_0", response_eval]
    stim_analysis_probed = stim_analysis_probed[interesting_cols]
    """
    stim_analysis_probed = stim_analysis_probed.drop(['versionString', 'versionDate', 'versionSettings', 'timeMS',
                                                      'timeMS_NoPauses', 'fullLogState', 'indexWithinFullLogs',
                                                      'stimID', 'probeIndexWithinDetails', 'isProbed', 'offsetTS',
                                                      'offsetTS_NoPauses', 'probeTS', 'probeTS_NoPauses', 'response',
                                                      'responseTS', 'responseTS_NoPauses', 'onset_averageDifficulty',
                                                      'response_averageDifficulty', 'avgNumFallingEssenses_m1000_p500',
                                                      'avgNumOppositeColorEssenses_m1000_p500',
                                                      'avgNumSameColorEssenses_m1000_p500',
                                                      'numButtonPresses_m1000_p500', 'numSaccades_m2000_p500',
                                                      'tempProxOfLastSacada', 'spatialProxFromStimLocToLastSacadaLine',
                                                      'averageGaze_m2000_p500', 'numBlinks_m2000_p500',
                                                      'tempProxOfLastBlink', 'subjectID', 'world', 'level', 'stimType',
                                                      'stimName'], axis=1)
    """

    if drop_columns is not None:
        stim_analysis_probed = stim_analysis_probed.drop(drop_columns, axis=1)
    else:
        drop_columns = []

    # for our Y to be classified correctly we need to convert it to numbers
    # make sure we don't have lines with odd responses
    stim_analysis_probed = stim_analysis_probed[stim_analysis_probed[response_eval].isin([FN, FP, TN, TP])]
    # convert categories to numbers
    stim_vis_dict = {FN: 0, FP: 1, TN: 2, TP: 3}
    stim_analysis_probed = stim_analysis_probed.replace(stim_vis_dict)
    pd.to_numeric(stim_analysis_probed[response_eval])
    # we are ONLY interested in SEEN vs UNSEEN i.e TP vs FN i.e 3 vs 0
    stim_analysis_probed = stim_analysis_probed[(stim_analysis_probed[response_eval] == 3) | (stim_analysis_probed[response_eval] == 0)]

    # encode categorical params as one-hot for the classifier to work properly
    """
    categorical_columns = ['stimDirection']  # this is the ONLY categorical parameter
    for col in categorical_columns:
        if col in drop_columns:  # to make sure we are not looking for a dropped categorical var
            continue
        stim_analysis_probed = pd.concat([stim_analysis_probed, pd.get_dummies(stim_analysis_probed[col], prefix=col)], axis=1)
        stim_analysis_probed.drop([col], axis=1, inplace=True)
    """

    # we will use support vector classification (SVC)
    # STEP 1: SPLIT the data to train and test, using
    # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    X = stim_analysis_probed.drop([response_eval], axis=1, inplace=False)  # the data w/o the response we want to predict
    y = stim_analysis_probed[response_eval]  # the response we want to predict

    # test_size is the % of the samples that we'll use for testing
    x_train, test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    """
    We normalize the X parameters in order for them to be in the same scale.
    Min-Max scaling (which leaves every parameter in a 1-0 range), seems to be the 
    frequent way to do so in ML:
    https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
    So we will use https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    Another option is https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    """
    if normalize:
        scaling_svm = Pipeline([('scaler', MinMaxScaler()),
                                ('svm', svm.LinearSVC(dual=False))])
        minmax_scaler = MinMaxScaler()
        x_train = pd.DataFrame(minmax_scaler.fit_transform(x_train))
        test = pd.DataFrame(minmax_scaler.transform(test))

    # classification with different Kernel functions
    if kernel == 'linear':
        train = svm.LinearSVC(dual=False)  # dual=False is preferred when n_samples > n_features
    else:
        train = svm.SVC(kernel=kernel)

    # Permutation test score
    """
    permutation_test_score generates a null distribution by calculating the accuracy of the classifier on 1000 different
    permutations of the dataset, where features remain the same but labels undergo different permutations. This is the 
    distribution for the null hypothesis which states there is no dependency between the features and labels. 
    An empirical p-value is then calculated as the percentage of permutations for which the score obtained is greater 
    that the score obtained using the original data.
    The p-value, which approximates the probability that the score would be obtained by chance. This is calculated as:
    (C + 1) / (n_permutations + 1) Where C is the number of permutations whose score >= the true score.
    The best possible p-value is 1/(n_permutations + 1), the worst is 1.0.
    
    StratifiedKfold generates test sets such that all contain the same distribution of classes, or as close as possible.
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    
    Balanced accuracy score 
    https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score
    """
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    n_uncorrelated_features = X.shape[1]
    rng = np.random.RandomState(seed=0)
    # Use same number of samples as in iris and the same number of features X.shape[1]
    X_rand = rng.normal(size=(X.shape[0], n_uncorrelated_features))

    print(f"{kernel} SVC Original Data")
    if normalize:
        score_data, perm_scores_data, pvalue_data = permutation_test_score(scaling_svm, X, y, scoring=scoring_method, cv=cv,
                                                                           n_permutations=n_permutations)
    else:
        score_data, perm_scores_data, pvalue_data = permutation_test_score(train, X, y, scoring=scoring_method, cv=cv,
                                                                           n_permutations=n_permutations)

    # classification with different Kernel functions
    if kernel == 'linear':
        train = svm.LinearSVC(dual=False)  # dual=False is preferred when n_samples > n_features
    else:
        train = svm.SVC(kernel=kernel)

    if normalize:
        scaling_random_svm = Pipeline([('scaler', MinMaxScaler()),
                                ('svm', svm.LinearSVC(dual=False))])

    print(f"{kernel} SVC Random Data")
    if normalize:
        score_rand, perm_scores_rand, pvalue_rand = permutation_test_score(scaling_random_svm, X_rand, y, scoring=scoring_method, cv=cv,
                                                                           n_permutations=n_permutations)
    else:
        score_rand, perm_scores_rand, pvalue_rand = permutation_test_score(train, X_rand, y, scoring=scoring_method, cv=cv,
                                                                           n_permutations=n_permutations)

    # these are the results of the model on original and random data
    result = pd.DataFrame({f"{kernel}_score_data": [score_data], f"{kernel}_score_data_pvalue": [pvalue_data],
                           f"{kernel}_score_random": [score_rand],
                           f"{kernel}_score_random_pvalue": [pvalue_rand]})
    result.to_csv(os.path.join(features_folder, f"{kernel}_permutation_test_{name}_{normalize}.csv"))
    # this is for a histogram of the permutations to plot the model's performance compared to the distribution
    permutations = pd.DataFrame({f"{kernel}_permutation_scores_data": perm_scores_data,
                                 f"{kernel}_permutation_scores_rand": perm_scores_rand})
    # plot: histogram of the permutation scores (the null distribution).
    # The line indicates the score obtained by the classifier on the original data.
    boxplotter.plot_histogram(data=permutations, column_name=f"{kernel}_permutation_scores_data",
                              plot_title=f"Original Data SVM Permutations",
                              plot_x_label="Accuracy Score", plot_y_label="Percent",
                              save_path=features_folder, save_name=f"{kernel}_perm_data_{name}",
                              num_of_bins=20, hist_shape=True, color="darkgray", vertical_line=score_data,
                              vertical_line_color="deeppink", xmin=0.5, xmax=0.52, xstep=0.004)

    boxplotter.plot_histogram(data=permutations, column_name=f"{kernel}_permutation_scores_rand",
                              plot_title=f"Random Data SVM Permutations",
                              plot_x_label="Accuracy Score", plot_y_label="Percent",
                              save_path=features_folder, save_name=f"{kernel}_perm_rand_{name}",
                              num_of_bins=20, hist_shape=True, color="darkgray", vertical_line=score_rand,
                              vertical_line_color="deeppink", xmin=0.5, xmax=0.52, xstep=0.004)
    permutations.to_csv(os.path.join(features_folder, f"permutations_data_{name}_{normalize}.csv"))

    print("SVM Fit")
    fitted = train.fit(x_train, y_train)
    boxplotter.confusion_matrix(fitted, test, y_test, labels=[stim_vis_dict[FN], stim_vis_dict[TP]],
                                display_labels=[FN, TP], save_path=features_folder,
                                save_name=f"svm_{kernel}_confusion_mat_{name}_{normalize}")

    if kernel == 'linear':
        # for the LINEAR KERNEL, we can see the weights assigned to the features (coefficients in the primal problem).
        # This is only available in the case of a linear kernel.
        linear_top_features = pd.DataFrame(pd.Series(abs(fitted.coef_[0]), index=x_train.columns).nlargest(10),
                                           columns=['Feature_Weights']).T
        boxplotter.plot_barh(data=linear_top_features, plot_title="Linear Kernel SVM Feature Weights",
                             plot_x_label="Weight",
                             plot_y_label="Feature", save_path=features_folder,
                             save_name=f"linear_feat_wei_{name}")
        linear_top_features.to_csv(os.path.join(features_folder, f"linear_feat_wei_{name}_{normalize}.csv"))

    # STEP 2: PREDICT
    print("SVM Prediction")
    pred = fitted.predict(test)
    # and output confusion matrix and TPR FPR etc
    columns = ["Kernel", TN, FP, FN, TP, TN + RATE, FP + RATE, FN + RATE, TP + RATE]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    tnr = tn / (tn + fp)
    fpr = tn / (tn + fp)
    fnr = fn / (fn + tp)
    tpr = tp / (fn + tp)
    kernel_list = [kernel, tn, fp, fn, tp, tnr, fpr, fnr, tpr]
    pred_df = pd.DataFrame([kernel_list], columns=columns)
    pred_df.to_csv(os.path.join(features_folder, f"{kernel}_vg_confusionmat_{name}_{normalize}.csv"))

    # STEP 3: save the SVM features as a dataframe
    analysis_features = pd.DataFrame({"Feature": stim_analysis_probed.columns})
    analysis_features.to_csv(os.path.join(features_folder, f"{kernel}_vg_features_{name}_{normalize}.csv"))
    return


def test_categorical_relations(sub_df, param1, param2, save_path):
    # path to save results
    folder_path = data_saver.create_analysis(save_path)
    features_folder = os.path.join(folder_path, "vg_features")
    # chi square for testing relations between param1 and param2 (two CATEGORICAL variables)
    expected, observed, stat = stats.chi_squared(data_name=f"{param1} and {param2}", data=sub_df, col1=param1, col2=param2)
    expected.to_csv(os.path.join(features_folder, f"chisq_{param1}_{param2}_expected.csv"))
    observed.to_csv(os.path.join(features_folder, f"chisq_{param1}_{param2}_observed.csv"))
    stat.to_csv(os.path.join(features_folder, f"chisq_{param1}_{param2}_stat.csv"))


def ttest_vis_distance_relations(sub_df, dependent_var_cols, save_path):
    # path to save results
    folder_path = data_saver.create_analysis(save_path)
    features_folder = os.path.join(folder_path, "vg_features")

    # t-test to test relationship between seen/unseen and stimulus distance from the closest essence/essence cluster
    avg_per_sub_vis = sub_df.groupby(["subjectID", response_eval]).mean().reset_index()

    # prepare df for ttest
    all_ttests = list()
    for dependent_var in dependent_var_cols:
        resp_dict = {TP: list(), FN: list()}
        for sub in list(avg_per_sub_vis["subjectID"].unique()):
            sub_data = avg_per_sub_vis[avg_per_sub_vis["subjectID"] == sub]
            for resp in resp_dict.keys():
                resp_dict[resp].append(sub_data.loc[sub_data[response_eval] == resp, dependent_var].values[0])
        test_df = pd.DataFrame.from_dict(resp_dict)
        test_df.describe().to_csv(os.path.join(features_folder, f"{dependent_var[5:]}_summary.csv"))
        stat = stats.paired_t_test_pg(test_df, col_1=test_df.columns[0], col_2=test_df.columns[1], df=True)
        stat['dependent_var'] = dependent_var
        all_ttests.append(stat)
    tests = pd.concat(all_ttests)
    tests['pval_Bonferroni_rejectH0'], tests['pval_Bonferroni_corrected'] = stats.correct_multi_comp(pvals=list(tests["p-val"]), alpha=0.05, method="bonf")
    tests.to_csv(os.path.join(features_folder, f"{BF_TTEST_PAIRED}_vis_by_spat_prox.csv"))
    return


def anova_vis_distance_relations(sub_df, dependent_var_cols, save_path):
    folder_path = data_saver.create_analysis(save_path)
    features_folder = os.path.join(folder_path, "vg_features")
    avg_per_sub_vis = sub_df.groupby(["subjectID", response_eval]).mean().reset_index()
    stat, tukey = stats.repeated_measures_anova(data_name="features_vg", data=avg_per_sub_vis, columns=dependent_var_cols, id_col="subjectID")
    return stat, tukey


def vis_per_stim(sub_data_dict, game_0_replay_1):
    all_resps_list = list()
    for sub in sub_data_dict:
        # get the responses
        if game_0_replay_1 == 0:  # game
            responses = getattr(sub_data_dict[sub].full.SessDetails, "ProbeDet")
        else:
            sub_responses = getattr(sub_data_dict[sub].full.SessDetails, "LocalizerDetWithFillers")
            # first, take out "additional presses": double-presses after the first press was already registered
            sub_responses = sub_responses[sub_responses[response_eval] != IRREL_ADDITION].reset_index(drop=True, inplace=False)
            # then, classify certain "afterWindowPresses" as responses and disregard the rest
            localizer_data, after_window_presses = quality_checks.after_window_presses_classifier(sub_responses)
            # no need for fillers here - we are analyzing responses by STIMULI
            responses = localizer_data[localizer_data["type"] == "LOCALIZER_STIMULUS"]
        # get the counts
        relevant_cols = responses[['stimulusName', response_eval]]
        relevant_cols = relevant_cols.fillna("None")
        resp_per_stim = pd.pivot_table(relevant_cols, index='stimulusName', columns=response_eval, aggfunc=len, fill_value=0, dropna=False)
        all_resps_list.append(resp_per_stim)
    result = reduce(lambda x, y: x.add(y, fill_value=0), all_resps_list)
    return result


def responses_to_rates(row):
    stim_present = row[TP] + row[FN]
    stim_absent = row[TN] + row[FP]
    tpr = row[TP] / stim_present if stim_present != 0 else 0
    fnr = row[FN] / stim_present if stim_present != 0 else 0
    tnr = row[TN] / stim_absent if stim_absent != 0 else 0
    fpr = row[FP] / stim_absent if stim_absent != 0 else 0
    new_row = row.copy()
    new_row[TP] = tpr
    new_row[FN] = fnr
    new_row[TN] = tnr
    new_row[FP] = fpr
    return new_row


def analyze_vis_per_stim_id(sub_dict, save_path):
    # path to save results
    folder_path = data_saver.create_analysis(save_path)
    features_folder = os.path.join(folder_path, "vg_features")
    game_stim_response_types = vis_per_stim(sub_dict, game_0_replay_1=0)
    replay_stim_response_types = vis_per_stim(sub_dict, game_0_replay_1=1).astype('float')
    # Turn response types to response type RATES
    game_stim_response_rates = game_stim_response_types.apply(lambda row: responses_to_rates(row), axis=1)
    replay_stim_response_rates = replay_stim_response_types.apply(lambda row: responses_to_rates(row), axis=1)
    game_stim_response_rates.to_csv(os.path.join(features_folder, f"vis_by_stim_GAME.csv"))
    replay_stim_response_rates.to_csv(os.path.join(features_folder, f"vis_by_stim_REPLAY.csv"))
    return









