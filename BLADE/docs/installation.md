---
id: installation
title: Installation Guide
sidebar_position: 1
---

# Installation Guide

Welcome to the BLADE installation guide. This document will walk you through all necessary steps to install and set up BLADE on your system, including system requirements, installation methods, setting up dependencies, and testing the installation.

## System Requirements

### Hardware Requirements

BLADE can operate on a variety of systems depending on the data size:
- **Small-scale data**: Suitable for Binder (1 CPU, 2GB RAM on Google Cloud).
- **Large-scale data**: Performance improves significantly with higher computational power. Our recommendation for optimal performance is:
  - **CPU**: 40 threads (Xeon 2.60GHz or equivalent)
  - **Memory**: 128 GB RAM

BLADE’s Empirical Bayes procedure includes independent optimization steps that can be parallelized to leverage multiple CPUs for faster execution.

### OS Requirements

BLADE is primarily tested on Linux-based operating systems. We recommend:
- **Linux distributions**: CentOS 7 or Ubuntu 16.04.
  
> **Note**: While BLADE might work on other OS platforms (macOS, Windows with WSL), these setups may require additional configurations.

## Installation Steps

### 1. Cloning the Repository

Begin by cloning the latest version of the BLADE codebase from the GitHub repository:

```bash
git clone https://github.com/aryamaanbose/BLADE_tasks.git
cd BLADE_tasks
```

### 2. Creating a Conda Environment

To manage dependencies for BLADE, we recommend creating a Conda environment.

#### Step 1: Installing Conda

If you don’t already have Conda installed, you can download and install Miniconda by following these steps:

1. Go to the [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html).
2. Download the appropriate installer for your OS.
3. Follow the installation instructions for your operating system to set up Conda.

After installation, you can verify Conda by running:

```bash
conda --version
```

#### Step 2: Creating the Conda Environment

Ensure you are in the `BLADE_tasks` directory, where the `environment.yml` file is located, and create the environment using:

```bash
conda env create -f environment.yml
```

This command will create an environment named `snakemake` with all necessary dependencies for BLADE.

#### Step 3: Activating the Environment

To activate the environment, run:

```bash
conda activate snakemake
```

Once activated, you are ready to use BLADE within this environment.

#### Step 4: Updating the Environment

To keep your environment up-to-date, you can pull the latest changes from the repository and update the environment:

**Pull the latest changes:**

```bash
git pull origin main
```

**Update the Conda environment:**

```bash
conda env update -f environment.yml
```

This command will ensure your `snakemake` environment is updated with any new dependencies added to the `environment.yml` file.

### 3. Using Snakemake Pipelines

Snakemake is a workflow management system that allows you to run complex pipelines. Here’s how to set up and use Snakemake with BLADE.

#### Step 1: Running a Snakemake Workflow

1. **Set up your workflow:** Ensure that you have the necessary files and configurations in place in the repository.

2. **Run the Snakemake pipeline:**

    ```bash
    snakemake -s your_workflow.smk
    ```

    Replace `your_workflow.smk` with the specific Snakemake file you are using. This command will execute the pipeline defined in the Snakemake workflow file.

#### Step 2: Specifying Resources

Snakemake allows you to specify resources such as the number of cores:

```bash
snakemake -s your_workflow.smk --cores 4
```

Adjust the `--cores` option based on your available resources.

#### Step 3: Running Snakemake in Dry-Run Mode

To see the planned workflow without executing it, use the `--dry-run` option:

```bash
snakemake -s your_workflow.smk --dry-run
```

This will list all steps without running them, which is helpful for testing your workflow setup.

#### Step 4: Running on a Specific Node or GPU (Optional)

If you have access to GPU resources or a specific node, you can configure Snakemake to target these resources by adjusting the resource specifications within your workflow file or using additional Snakemake flags, if supported by your cluster.

## Testing the Installation

After completing the installation steps, it is recommended to run a test workflow to ensure that everything is set up correctly. Follow the instructions in the `tests` directory of the repository to execute sample pipelines and verify the installation.

## Troubleshooting

If you encounter any issues during the installation or setup process, please refer to the [Troubleshooting](./troubleshooting.md) section or open an issue on the [GitHub repository](https://github.com/aryamaanbose/BLADE_tasks/issues).

## Additional Resources

- [BLADE Documentation](https://github.com/aryamaanbose/BLADE_tasks/wiki)
- [Conda Documentation](https://docs.conda.io/)
- [Snakemake Documentation](https://snakemake.readthedocs.io/)

---
```