# MEG preprocessing

This directory contains the code developed by Oscar Ferrante to preprocess the MEG data from Experiment 2 of the COGITATE project.

## System requirements

The pipeline was developed and tested on Linux and on a high-performance computing (HPC) system. Windows and macOS are not currently supported or thoroughly tested, and platform-specific issues may occur.

## Installation

Create the Conda environment from the provided environment file:

```bash
conda env create --file requirements_cogitate_meg.yaml
```

The approximate installation time is **90 minutes**, although this will depend on the system and network connection.

## Sample data

Sample data for demonstrating the preprocessing pipeline are available from the [COGITATE data bundles webpage](https://www.arc-cogitate.com/data-bundles).

The data bundle includes:

- data converted to the Brain Imaging Data Structure (BIDS);
- preprocessed derivatives in `derivatives/preprocessing/`; and
- FreeSurfer derivatives in `derivatives/fs/`.

Download and extract the sample dataset before running the demo.

## Configuration

Set the BIDS root directory in:

```text
meeg/config/config.py
```

For the sample dataset, the configured path should point to:

```text
$ROOT/sample_data/bids
```

If required, also update the paths in the preprocessing scripts so that they point to the downloaded data.

## Running the preprocessing demo

The example below processes participant `SA124`.

From the command line, run preprocessing step 1:

```bash
python REPO_ROOT/cogitate-msp1/scripts/meeg/preprocessing/99_run_preproc.py \
    --sub SA124 \
    --visit V2 \
    --record run \
    --step 1
```

After step 1 has completed successfully, run step 2:

```bash
python REPO_ROOT/cogitate-msp1/scripts/meeg/preprocessing/P99_run_preproc.py \
    --sub SA124 \
    --visit V2 \
    --record run \
    --step 2
```

Replace `REPO_ROOT` with the path to the cloned repository.

## Expected output

The pipeline creates the following participant-level derivatives directory:

```text
$ROOT/sample_data/bids/derivatives/preprocessing/sub-SA124/
```

This directory contains separate subdirectories for the preprocessing stages. The files generated during epoching contain the final preprocessed data used as input for subsequent analyses.

The approximate runtime for the complete demo is **90 minutes**, although the exact duration will depend on the available hardware and computing resources.
