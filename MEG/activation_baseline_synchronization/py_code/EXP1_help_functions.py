'''
help function from EXP1
in .sh
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class BidsPath




'''
#%%
import os
import os.path as op
import scipy.signal as ss
import matplotlib.pyplot as plt
import numpy as np
import scipy
import mne
import mne_bids

# for create_inverse
from mne.cov import compute_covariance
# for filter data
import scipy.signal as ss

from EXP1_config import bids_root

#%%
class BidsPath:
    def __init__(self, subject_id, visit_id):
        # Store subject and visit ids
        self.subject_id = subject_id
        self.visit_id = visit_id
        
        # Set default paths and create the related folders
        self.prep_deriv_root = op.join(bids_root, "derivatives", "preprocessing")
        self.fwd_deriv_root = op.join(bids_root, "derivatives", "forward")
        self.fs_deriv_root = op.join(bids_root, "derivatives", "fs")
        self.ged_deriv_root = op.join(bids_root, "derivatives", "ged")
        self.roi_deriv_root = op.join(bids_root, "derivatives", "roilabel")
    
    def add_deriv(self, new_deriv_folder_name, deriv_path_handle, create_folder=True):
        # Add new derivatives path to the BidsPath object
        setattr(self, f"{deriv_path_handle}_deriv_root", op.join(
            bids_root, "derivatives", new_deriv_folder_name))
        
        # Create new derivatives folder
        if not op.exists(getattr(self, f"{deriv_path_handle}_deriv_root")):
            os.makedirs(getattr(self, f"{deriv_path_handle}_deriv_root"))
        
        # Create figure folder
        if create_folder:
            setattr(self, f"{deriv_path_handle}_figure_root", op.join(
                getattr(self, f"{deriv_path_handle}_deriv_root"),
                f"sub-{self.subject_id}", f"ses-{self.visit_id}", "meg", "figures"))
            if not op.exists(getattr(self, f"{deriv_path_handle}_figure_root")):
                os.makedirs(getattr(self, f"{deriv_path_handle}_figure_root"))

#%% crop_stcs
def crop_stcs(stcs, tmin=0., tmax=.5):
    stcs_act = []
    # Loop over epochs
    for i in range(len(stcs)):
        # Select active time window
        stcs_act.append(stcs[i].copy().crop(tmin, tmax))
    
    return stcs_act

#%% comp_cov_stcs
def comp_cov_stcs(stcs):
    """
    Compute trial-wise covariance matrices from source time courses.

    Parameters
    ----------
    stcs : list of mne.SourceEstimate
        Source time courses for each trial,
        shape per trial = (n_vertices, n_times).

    Returns
    -------
    cov : 
        Trial covariance matrices,
        list of length n_trials; each element shape (n_vertices, n_vertices)
    """
    
    cov = []                                  # list, will have length n_trials
    for stc in stcs:
        data = stc.data  # shape: (n_vertices, n_times)

        # Mean-center vertices
        data = data - np.mean(data, axis=1, keepdims=True)  # shape: (n_vertices, n_times)

        # Compute sample covariance across time
        # (n_vertices x n_times) @ (n_times x n_vertices) → (n_vertices, n_vertices)
        cov_trial = data @ data.T / (data.shape[1] - 1)

        # Append trial covariance
        cov.append(cov_trial)  # each shape: (n_vertices, n_vertices)
        
    return cov  # list, len = n_trials; each element (n_vertices, n_vertices)


#%% ged_clean_and_average_cov
def ged_clean_and_average_cov(cov):
    """
    Clean and average trial-wise covariance matrices.
    remove outlier trial covariances and return their average

    Parameters
    ----------
    cov : # list, len = n_trials; each element (n_vertices, n_vertices)
        Trial covariance matrices.

    Returns
    -------
    cov_avg : ndarray, shape (n_vertices, n_vertices)
        Average covariance matrix with outlier trials (>3 SD from mean) excluded.
    """
    
    # Average covariance over trials
    cov_m = np.mean(cov, axis=0)             # shape: (n_vertices, n_vertices)
    
    # Loop over trials
    # Euclidean distance of each trial covariance to the mean
    dists = []                                # list, will have length n_trials
    for i in range(len(cov)):
        tcov = cov[i]                         # shape: (n_vertices, n_vertices)
    
        # Flatten both trial covariance and mean covariance
        tcov_flat = tcov.reshape(1, -1)       # shape: (1, n_vertices*n_vertices)
        cov_m_flat = cov_m.reshape(1, -1)     # shape: (1, n_vertices*n_vertices)
        
        # Difference between trial and mean
        diff = tcov_flat - cov_m_flat         # shape: (1, n_vertices*n_vertices)
        
        # Squared difference
        sq_diff = diff**2                     # shape: (1, n_vertices*n_vertices)
        
        # Sum over all entries
        sum_sq = np.sum(sq_diff)              # shape: scalar
        
        # Square root → Euclidean (Frobenius) distance
        dist = np.sqrt(sum_sq)                # shape: scalar
        
        # Append result
        dists.append(dist)                    # after loop: list of length n_trials
        
    # Compute z-scored distance
    dists_Z = (dists-np.mean(dists)) / np.std(dists) # shape: (n_trials,)
    
    # Average trial-covariances together, excluding outliers (>3 SD)
    # shape: (n_vertices, n_vertices)
    cov_avg = np.mean( np.asarray(cov)[dists_Z<3] ,axis=0)
    
    return cov_avg

#%% ged_compute
def ged_compute(covAm, covBm, desc, bids_task, bids_paths, save=True):
    # Run GED
    evals,evecs = scipy.linalg.eigh(covAm,covBm)
    
    # Sort eigenvalues/vectors
    sidx  = np.argsort(evals)[::-1]
    evals = evals[sidx]
    evecs = evecs[:,sidx]
    
    # Save results
    if save:
        bids_path_ged = mne_bids.BIDSPath(
            root=bids_paths.ged_deriv_root, 
            subject=bids_paths.subject_id,  
            datatype='meg',  
            task=bids_task,
            session=bids_paths.visit_id, 
            suffix=f'desc-{desc}_evals',
            extension='.npy',
            check=False)
        np.save(bids_path_ged.fpath, evals)
        
        bids_path_ged = bids_path_ged.copy().update(
            suffix=f'desc-{desc}_evecs',)
        np.save(bids_path_ged.fpath, evecs)
    
    return evals, evecs
#%% ged_compute
def ged_compute(covAm, covBm, desc, bids_task, bids_paths, save=True):
    """
    Compute Generalized Eigendecomposition (GED) and save the results.

    Parameters
    ----------
    covAm : ndarray, shape (n_vertices, n_vertices)
        The first covariance matrix (e.g., from an active condition).
    covBm : ndarray, shape (n_vertices, n_vertices)
        The second covariance matrix (e.g., from a reference or baseline condition).
    desc : str
        A brief description for the output file (e.g., 'alpha').
    bids_task : str
        The BIDS task name.
    bids_paths : object
        An object containing BIDS-related path information (e.g., root, subject_id).
    save : bool, optional
        Whether to save the eigenvalues and eigenvectors. Defaults to True.

    Returns
    -------
    evals : ndarray, shape (n_vertices,)
        The eigenvalues of the GED, sorted in descending order.
    evecs : ndarray, shape (n_vertices, n_vertices)
        The eigenvectors (spatial filters) of the GED, sorted to correspond with the eigenvalues.
    """
    # Perform Generalized Eigendecomposition (GED) on the two covariance matrices.
    # The function solves the generalized eigenvalue problem: covAm * v = lambda * covBm * v
    # evals: The eigenvalues (lambda), representing the power ratio of covAm to covBm.  shape (n_vertices,)
    # evecs: The eigenvectors (v), which are the spatial filters or components. shape (n_vertices, n_vertices)
    evals, evecs = scipy.linalg.eigh(covAm, covBm)

    # Sort eigenvalues and eigenvectors together to preserve their correspondence.
    # np.argsort returns the indices that would sort the array. [::-1] reverses them for descending order.
    # This places the most prominent components (those with the highest power in covAm relative to covBm) at the beginning of the arrays.
    sidx = np.argsort(evals)[::-1] # sidx: shape (n_vertices,)
    
    # Use the sorted indices to reorder the eigenvalues.
    evals = evals[sidx] # evals: shape (n_vertices,)
    
    # Use the same sorted indices to reorder the columns of the eigenvectors.
    evecs = evecs[:, sidx]# evecs: shape (n_vertices, n_vertices)
    
    # Save the results to BIDS-compliant paths.
    if save:
        bids_path_ged = mne_bids.BIDSPath(
            root=bids_paths.ged_deriv_root, 
            subject=bids_paths.subject_id,  
            datatype='meg',  
            task=bids_task,
            session=bids_paths.visit_id, 
            suffix=f'desc-{desc}_evals',
            extension='.npy',
            check=False)
        np.save(bids_path_ged.fpath, evals)
        
        bids_path_ged = bids_path_ged.copy().update(
            suffix=f'desc-{desc}_evecs',)
        np.save(bids_path_ged.fpath, evecs)
    
    return evals, evecs

#%% ged_apply_reg
def ged_apply_reg(cov):
    """
    Applies Tikhonov regularization to a covariance matrix.

    This process adds a small amount of stability to the matrix, preventing
    numerical errors that can occur during subsequent calculations like
    Generalized Eigendecomposition (GED).

    Parameters
    ----------
    cov : list of lenght of  ntrial, shape (n_vertices, n_vertices)
        The input covariance matrix.

    Returns
    -------
    cov_r : ndarray, shape (n_vertices, n_vertices)
        The regularized (stabilized) covariance matrix.
    """
    
    # Define the regularization parameter (gamma). A small value like 0.01
    # means a small amount of stability is added.
    gamma = 0.01
    
    # Step 1: Find both the eigenvalues and eigenvectors of the original matrix.
    # The scipy.linalg.eigh function returns a tuple:
    # - The first item (index 0) is an array of eigenvalues.
    # - The second item (index 1) is a 2D array of eigenvectors.
    eigenvalues, eigenvectors = scipy.linalg.eigh(cov)
    
    # Step 2: Calculate the average of the eigenvalues.
    # This is used to properly scale the regularization term
    # use the 'eigenvalues' variable from the previous step.
    mean_eigenvalue = np.mean(eigenvalues)
    
    # Step 3: Create an identity matrix with the same size as the covariance matrix.
    # An identity matrix has 1 on the main diagonal and 0 everywhere else.
    identity_matrix = np.eye(len(cov))
    
    # Step 4: Apply the regularization formula.
    # The formula combines the original matrix (scaled by 1-gamma) with the scaled identity matrix.
    # This adds a small, uniform value to the diagonal, making the matrix more stable.
    cov_r = cov * (1 - gamma) + gamma * mean_eigenvalue * identity_matrix

  
    return cov_r

#%% get_ged
def get_ged(stcs_cond_act, stcs_nocond_act, condition, bids_task, label_name, bids_paths, save=True):
    """
    Computes Generalized Eigendecomposition (GED) for a specific condition against a reference,
    using source time courses.

    computes trial-wise covariances, cleans and averages them, applies regularization, and finally 
    computes and returns the GED results.

    Parameters
    ----------
    stcs_cond_act : list of mne.SourceEstimate
        Source time courses for the condition of interest.
    stcs_nocond_act : list of mne.SourceEstimate
        Source time courses for the non-condition (reference).
    condition : str
        The name of the condition (e.g., 'alpha', 'beta').
    bids_task : str
        The BIDS task name.
    label_name : str
        Name of the source region (e.g., 'frontal_lobe').
    bids_paths : object
        An object containing BIDS-related path information.
    save : bool, optional
        Whether to save the GED results. Defaults to True.

    Returns
    -------
    evals_cond : ndarray, shape (n_vertices,)
        Sorted eigenvalues from the GED.
    evecs_cond : ndarray, shape (n_vertices, n_vertices)
        Sorted eigenvectors from the GED.
    """
    
    # --- Step 1: Compute trial-wise covariance matrices ---
    # `comp_cov_stcs` computes a covariance matrix for each trial.
    # cov_cond  : list of length n_trials; each element (n_vertices, n_vertices)
    # cov_nocond: list of length n_trials; each element (n_vertices, n_vertices)
    cov_cond = comp_cov_stcs(stcs_cond_act)
    cov_nocond = comp_cov_stcs(stcs_nocond_act)
    
    # --- Step 2: Clean and average covariance matrices ---
    # `ged_clean_and_average_cov` removes outlier trials and averages the
    # remaining covariance matrices. This results in a single, representative
    # covariance matrix for each condition.
    # cov_cond: ndarray, shape (n_vertices, n_vertices)
    # cov_nocond: ndarray, shape (n_vertices, n_vertices)
    cov_cond = ged_clean_and_average_cov(cov_cond)
    cov_nocond = ged_clean_and_average_cov(cov_nocond)
    
    # --- Step 3: Apply regularization to the reference covariance matrix ---
    # `ged_apply_reg` adds a small amount of identity matrix to the reference
    # covariance matrix to prevent numerical instability during GED.
    # cov_nocond: ndarray, shape (n_vertices, n_vertices)
    cov_nocond = ged_apply_reg(cov_nocond)
    
    # --- Step 4: Run Generalized Eigendecomposition (GED) ---
    # `ged_compute` finds the spatial filters (eigenvectors) that maximize
    # the power in the condition of interest (cov_cond) relative to the reference (cov_nocond).
    # evals_cond: ndarray, shape (n_vertices,)
    # evecs_cond: ndarray, shape (n_vertices, n_vertices)
    evals_cond, evecs_cond = ged_compute(
        cov_cond, cov_nocond, 
        f"{label_name},{condition}", bids_task, bids_paths, save=save)
    
    # Xuan add: no plot, no save
    # # Plot covariance
    # plot_cov(
    #     cov_cond, cov_nocond, 
    #     f"task-{bids_task}_desc-{label_name}_{condition}", bids_paths, save=save)

    # # Plot GED eigenvalues
    # ged_plot_evals(
    #     evals_cond, 
    #     f"task-{bids_task}_desc-{label_name}_{condition}", bids_paths, save=save)
    
    # # Compute GED spatial filter
    # ged_create_spatial_filter(
    #     evecs_cond, 
    #     f"{label_name},{condition}", bids_task, bids_paths, save=save)
    
    return evals_cond, evecs_cond

# #%% original code: get_ged
# def get_ged(stcs_cond_act, stcs_nocond_act, condition, bids_task, label_name, bids_paths, save=True,):
#     # Compute covariance
#     cov_cond = comp_cov_stcs(
#         stcs_cond_act)
#     cov_nocond = comp_cov_stcs(
#         stcs_nocond_act)
    
#     # Remove outliers and average
#     cov_cond = ged_clean_and_average_cov(
#         cov_cond)
#     cov_nocond = ged_clean_and_average_cov(
#         cov_nocond)
    
#     # Apply regularization
#     cov_nocond = ged_apply_reg(
#         cov_nocond)
    

#     # Run GED
#     evals_cond, evecs_cond = ged_compute(
#         cov_cond, cov_nocond, 
#         f"{label_name},{condition}", bids_task, bids_paths, save=save)

    

#%% 
def ged_get_time_course(stcs, 
                        evecs, 
                        desc, 
                        bids_task, 
                        bids_paths, 
                        prep_erf=True, 
                        save=True):
    comp_ts = []
    print('GED using', len(stcs), ' stc')
    try:
        # Loop over epochs
       
        for i in range(len(stcs)):
            
            # Get data
            data = stcs[i].data
            times = stcs[i].times
            
            # Apply GED filter
            comp_ts.append(evecs[:,0].T @ data)
    except:
        # Get data
        data = stcs.data # Xuan add:shape (n_dipoles, n_times)
        times = stcs.times
        
        # Apply GED filter
        comp_ts.append(evecs[:,0].T @ data)
        
    # Xuan add: not plot
    # # Preprocess ERF
    # if prep_erf:
    #     comp_ts = prep_erf_results(
    #         comp_ts, times=times)
    
    # Save results
    if save:
        bids_path_ged = mne_bids.BIDSPath(
            root=bids_paths.ged_deriv_root, 
            subject=bids_paths.subject_id,  
            datatype='meg',  
            task=bids_task,
            session=bids_paths.visit_id, 
            suffix=f'desc-{desc}_compts',
            extension='.npy',
            check=False)
        
        np.save(bids_path_ged.fpath, comp_ts)
    
    return comp_ts

#%% list_filetype
def list_filetype(directory, extension=".json"):
    try:
        json_files = [os.path.join(directory,f) for f in os.listdir(directory) if f.endswith(extension)]
        return json_files
    except FileNotFoundError:
        return f"Error: {directory} is not a valid directory."
#%% set path
def set_task(visit_id, bids_task_v2=None, is_rest=False):
    # Set task
    if is_rest:
        bids_task = "rest"
    elif visit_id == "V1":
        bids_task = 'dur'
    elif visit_id == "V2":
        bids_task = bids_task_v2
    else:
        raise ValueError("Error: could not set the task")
    
    return bids_task

# %% set_bids_path
def set_bids_path(subject_id, visit_id, bids_path, bids_task_v2=None, is_rest=False):
    # Set task
    bids_task = set_task(visit_id, bids_task_v2, is_rest)
    
    # Set bids path
    bids_path = mne_bids.BIDSPath(
        root=bids_path, 
        subject=subject_id,  
        datatype='meg',  
        task=bids_task,
        session=visit_id, 
        suffix='epo',
        extension='.fif',
        check=False)
    
    return bids_path
# %% read_epochs
def read_epochs(subject_id, visit_id, bids_task_v2=None, preload=False, pick_meg_only=True, is_rest=False, debug=False):
    # Set data paths
    bids_paths = BidsPath(subject_id, visit_id)
    print(f"Loading epochs for subject {subject_id}")
    
    # Set bids path
    bids_path_epo = set_bids_path(
        subject_id, visit_id, 
        bids_paths.prep_deriv_root, 
        bids_task_v2,
        is_rest=is_rest)
    
    # Read epoched data
    epochs = mne.read_epochs(
        bids_path_epo.fpath,
        preload=preload)
    
    # If debug is True, only load the first 100 epochs
    if debug:
        epochs = epochs[0:30]
    
    # Select sensor type
    if pick_meg_only:
        epochs.load_data().pick('meg')
    
    return epochs

# %% read_fwd
def read_fwd(subject_id, 
             visit_id, 
             bids_task_v2=None, 
             inv_method="dspm"):
    # Set data paths
    bids_paths = BidsPath(subject_id, visit_id)
    
    # Set space
    if inv_method == 'dspm':
        space = "surface"
    else:
        space = "volume"
    
    # Set task
    if visit_id == "V1":
        bids_task = None
    elif visit_id == "V2":
        bids_task = set_task(visit_id, bids_task_v2)
    
    # Set bids path
    bids_path_fwd = mne_bids.BIDSPath(
        root=bids_paths.fwd_deriv_root, 
        subject=subject_id,  
        datatype='meg',  
        task=bids_task,
        session=visit_id, 
        suffix=space+"_fwd",
        extension='.fif',
        check=False)
    
    # Read forward model
    fwd = mne.read_forward_solution(bids_path_fwd.fpath)
    
    return fwd

# %%  create_inverse
def create_inverse(subject_id, visit_id, epochs, fwd, 
                   fr_band=None, use_rs_noise=False, epochs_rs=None, 
                   active_win=(.0, .5), baseline_win=(-0.5, 0.), plot_cov=False):
    # Compute rank
    rank = mne.compute_rank(epochs, 
                            tol=1e-6, 
                            tol_kind='relative')
    if visit_id == "V2" and use_rs_noise:
        rank_rs = mne.compute_rank(epochs_rs, 
                                   tol=1e-6, 
                                   tol_kind='relative')
        if rank_rs['meg'] < rank['meg']:
            rank = rank_rs
    
    # Filter data
    if fr_band:
        if fr_band == "alpha":
            fmin = 8
            fmax = 13
            # bandwidth = 2.
        elif fr_band == "gamma":
            fmin = 60
            fmax = 90
            # bandwidth = 4.
        else:
            raise ValueError("Error: 'band' value not valid")
        
        epochs = epochs.filter(fmin, fmax)
        if visit_id == "V2" and use_rs_noise:
            epochs_rs = epochs_rs.filter(fmin, fmax)
    
    # Compute covariance matrices
    if visit_id == "V1" or not use_rs_noise:
        noise_cov = compute_covariance(epochs, 
                                       tmin=baseline_win[0], 
                                       tmax=baseline_win[1], 
                                       method='empirical', 
                                       rank=rank)
    elif visit_id == "V2" and use_rs_noise:
        noise_cov = compute_covariance(epochs_rs, 
                                       method='empirical', 
                                       rank=rank)
    
    active_cov = compute_covariance(epochs, 
                                    tmin=active_win[0], 
                                    tmax=active_win[1],
                                    method='empirical', 
                                    rank=rank)
    common_cov = noise_cov + active_cov
    
    # Plot covariace matrices
    if plot_cov:
        # Set data paths
        bids_paths = BidsPath(subject_id, visit_id)
        bids_paths.add_deriv( "source_loc", "souloc")
        
        # Plot
        fname_fig = op.join(bids_paths.souloc_figure_root,
                            "covariance_%s_%s.png")
        if visit_id == "V1" or not use_rs_noise:
            fig = noise_cov.plot(info=epochs.info)
        elif visit_id == "V2":
            fig = noise_cov.plot(info=epochs_rs.info)
        fig[0].savefig(fname_fig % ("noise", "cov"))
        fig[1].savefig(fname_fig % ("noise", "rank"))
        
        fig = active_cov.plot(info=epochs.info)
        fig[0].savefig(fname_fig % ("active", "cov"))
        fig[1].savefig(fname_fig % ("active", "rank"))
        plt.close("all")
    
    # Make inverse operator 

    inverse = mne.minimum_norm.make_inverse_operator(
        epochs.info,
        fwd, 
        common_cov,
        loose=.2,# If 0, then the solution is computed with fixed orientation; if 1  the solution is computed with free orientations. Value that weights the source variances of the dipole components that are parallel (tangential) to the cortical surface. 
        depth=.8,
        fixed=False,# Not Use fixed source orientations normal to the cortical mantle.
        rank=rank,
        use_cps=True # use cortical patch statistics to define normal orientations for surfaces (default True).
        )
    
    # Prepare covs variable to return
    covs = {"noise_cov":noise_cov, "active_cov":active_cov}
    
    return inverse, rank, covs
# %% get_labels_details
def get_labels_details(labels):
    # Organise labels for cortical parcellation
    n_labels = len(labels)
    label_colors = [label.color for label in labels]
    # First, we reorder the labels based on their location in the left hemi
    label_names = [label.name for label in labels]
    lh_labels = [name for name in label_names if name.endswith('lh')]
    
    # Get the y-location of the label
    label_ypos = list()
    for name in lh_labels:
        idx = label_names.index(name)
        ypos = np.mean(labels[idx].pos[:, 1])
        label_ypos.append(ypos)
    
    # Reorder the labels based on their location
    lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]
    
    # For the right hemi
    rh_labels = [label[:-2] + 'rh' for label in lh_labels]
    
    return lh_labels, rh_labels, label_names, n_labels, label_colors
# %% lowpass_filter
def lowpass_filter(data, order, filt_type, cutoff, fs=1000.):
    if filt_type == "fir":
        # Low-pass filter the data using an FIR filter
        b = ss.firwin(
            order * 2 + 1, 
            cutoff / (0.5 * fs), 
            window='hamming')
        a = [1.]
    elif filt_type == "butter":
        # Low-pass filter the data using butterworth filter
        b, a = ss.butter(
            order, 
            cutoff, 
            fs=fs, 
            btype='low', 
            analog=False)
    
    data_filt = ss.lfilter(b, a, data)
    
    return data_filt