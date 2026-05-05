# Reproducibility Code: Bentkus-type asymptotic e-values

This repository contains the Python scripts to reproduce the empirical simulations for the paper: **"Bentkus-type asymptotic e-values."**

The code is divided into three distinct files:
1. `tau_evaluation.py`: Evaluates and inverts the function tau defined in the paper. 
2. `bentkus_evariable.py`: Plots the Bentkus threshold functions. 
2. `mixtures_experiments.py`: Runs the multiple testing and posthoc experiments and saves the tables.s

## Requirements

The simulations are written in standard Python and rely on common scientific computing libraries. To install the required dependencies, run:

```bash
pip install numpy pandas matplotlib scipy tqdm