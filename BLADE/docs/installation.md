---
id: installation
title: Installation Guide
sidebar_position: 1
---

# Installation

## System Requirements

### Hardware Requirements

BLADE can run on minimal computer specifications, such as Binder (1 CPU, 2GB RAM on Google Cloud), when the data size is small. However, BLADE can significantly benefit from a larger number of CPUs and RAM. The Empirical Bayes procedure of BLADE runs independent optimization procedures that can be parallelized. In our evaluation, we used a computing node with the following specifications:
- 40 threads (Xeon 2.60GHz)
- 128 GB RAM

### OS Requirements

The package development version is tested on Linux operating systems (CentOS 7 and Ubuntu 16.04).

## Installation Methods

### Using pip

The python package of BLADE is available on pip. You can install it simply by running (takes only less than 1 minute):

```bash
pip install BLADE_Deconvolution
