# MEG source modelling

This directory contains the code developed by Oscar Ferrante for source modelling of the MEG data from Experiment 2 of the COGITATE project.

## System requirements

The source-modelling pipeline was developed and tested on Linux and on a high-performance computing (HPC) system. Windows and macOS are not currently supported or thoroughly tested, and platform-specific issues may occur.

The pipeline requires:

- [Conda](https://docs.conda.io/);
- [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/); and
- an individual anatomical MRI scan processed with FreeSurfer, or the FreeSurfer `fsaverage` template when an individual MRI scan is unavailable.

## Installation

### Python environment

Create the Conda environment from the provided environment file:

```bash
conda env create --file requirements_cogitate_meg.yaml
```

The approximate installation time for the Conda environment is **90 minutes**, although this will depend on the system and network connection.

### FreeSurfer

Install and configure FreeSurfer by following the instructions on the [FreeSurfer website](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall).

## Sample data

Sample data for demonstrating the analysis pipeline are available from the [Cogitate website](https://www.arc-cogitate.com/data-bundles).

The bundle contains MEG data from a small subset of participants. It includes:

- data converted to the Brain Imaging Data Structure (BIDS);
- preprocessed derivatives in `derivatives/preprocessing/`; and
- FreeSurfer derivatives in `derivatives/fs/`.

Download and extract the sample dataset before running the demo.

## Configuration

Set the BIDS root directory in:

```text
config.py
```

For the sample dataset, the configured path should point to:

```text
$ROOT/sample_data/bids
```

If required, update the paths in the source-modelling scripts and SLURM submission scripts so that they point to the downloaded BIDS and FreeSurfer data.

## Running the source-modelling pipeline

The steps below use participant `SA124` as an example. Run them in the order shown.

The SLURM scripts construct the participant identifier by combining the prefix supplied on the command line with the zero-padded array-task identifier. For example, `SA` and array task `124` produce participant `SA124`.

### 1. Process the anatomical MRI with FreeSurfer, if required

Skip this step if a completed FreeSurfer reconstruction is already available in `SUBJECTS_DIR` or if the supplied FreeSurfer derivatives are being used.

To process participant `SA124` using the provided SLURM script, run:

```bash
sbatch --array=124 srun_freesurfer.sh SA V2
```

Under the current implementation, `srun_freesurfer.sh` expects two positional arguments, although the visit argument is not used by the `recon-all` command. The script runs FreeSurfer `recon-all` and writes the reconstruction to the directory specified by `SUBJECTS_DIR`.

Before submission, edit the hard-coded values of `#SBATCH --chdir`, `SUBJECTS_DIR`, `nifti_dir`, the FreeSurfer module, and any site-specific HPC settings.

### 2. Construct the boundary-element model

Run `S00_bem.py` to construct the surfaces and boundary-element model (BEM) required for forward modelling.

Using the SLURM submission script:

```bash
sbatch --array=124 srun_bem.sh SA V1
```

Alternatively, run the Python script directly:

```bash
python REPO_ROOT/cogitate-msp2/MEG/source_modelling/S00_bem.py \
    --sub SA124 \
    --visit V1
```

Replace `REPO_ROOT` with the path to the cloned repository.

### 3. Co-register the MEG and anatomical MRI data

Co-registration aligns the MEG sensor geometry and digitised head shape with the participant's anatomical MRI. We recommend performing and checking this step manually with the [MNE-Python co-registration interface](https://mne.tools/stable/generated/mne.gui.coregistration.html):

```bash
mne coreg
```

In the graphical interface:

1. select the participant's FreeSurfer reconstruction;
2. load the corresponding MEG measurement file;
3. align the fiducial points;
4. refine the fit using the digitised head-shape points;
5. inspect the alignment from multiple views; and
6. save the resulting MEG–MRI transformation file in the location and naming format expected by the forward-modelling script.

Carefully verify the co-registration before continuing, because alignment errors directly affect the anatomical accuracy of the source estimates.

### 4. Compute the forward model

If an individual anatomical MRI and FreeSurfer reconstruction are available, use the provided SLURM script:

```bash
sbatch --array=124 srun_forward.sh SA V1 surface
```

Here, `surface` is passed to the Python script through the `--space` argument.

Alternatively, run the Python script directly:

```bash
python REPO_ROOT/cogitate-msp2/MEG/source_modelling/S01_forward_model.py \
    --sub SA124 \
    --visit V2 \
    --space surface
```

If no individual anatomical MRI is available, use the template-based workflow instead:

```bash
python REPO_ROOT/cogitate-msp2/MEG/source_modelling/S01b_forward_model_template.py \
    --sub SA124 \
    --visit V2
```

Use either `S01_forward_model.py` or `S01b_forward_model_template.py`, depending on MRI availability; the two scripts are alternative workflows.

## Running multiple participants on SLURM

Supply the participant prefix as the first argument and participant numbers through the SLURM array option. For example:

```bash
sbatch --array=101,103,105 srun_bem.sh SA V1
sbatch --array=101,103,105 srun_forward.sh SA V1 surface
```

These commands process participants `SA101`, `SA103`, and `SA105`. Submit the forward-modelling jobs only after the corresponding BEM and co-registration steps have been completed successfully.

## Expected output

The pipeline generates the participant-level source-modelling derivatives required by subsequent source-space analyses. Depending on the selected workflow, these outputs include:

- a FreeSurfer reconstruction, if `recon-all` was required;
- BEM surfaces and solutions;
- the MEG–MRI transformation produced during co-registration;
- the source space; and
- the forward solution.

Before proceeding to source reconstruction, verify that the expected files were created and that the forward solution uses the intended participant-specific or template anatomy.

## Approximate runtime

The complete individual-level source-modelling workflow takes approximately **210 minutes per participant**, excluding FreeSurfer reconstruction and manual co-registration. FreeSurfer `recon-all` may require several hours, depending on the anatomical image and available hardware. Subsequent group-level source-modelling analyses take approximately **60 minutes**.
