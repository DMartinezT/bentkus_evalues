# Bentkus-type asymptotic e-values

Python code for the numerical results in **Bentkus-type asymptotic e-values**.
All experiments use simulated data and run on a CPU.

## Setup

The experiments were verified with Python 3.8.5 and the pinned dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the following commands from this directory. Scripts resolve their table
output paths relative to their own location.

## Manuscript results

| File | Result |
| --- | --- |
| `bentkus_evariable.ipynb` | Both threshold-comparison figures in `plots/` |
| `tau_evaluation.ipynb` | The boundary in alpha for a target level of 0.1 |
| `mixtures_experiments.py` | Post-hoc inference and Gaussian multiple-testing tables |
| `heavytail_experiment.py` | Heavy-tailed raw-sample multiple-testing table and Monte Carlo standard errors |
| `grid_penalties.py` | Finite-grid mixing penalties for the experimental mixtures |
| `verify_reproduction.py` | Numerical checks and agreement with the manuscript heavy-tail table |

```bash
python3 mixtures_experiments.py
python3 heavytail_experiment.py
python3 grid_penalties.py
python3 verify_reproduction.py
```

The Gaussian and post-hoc script writes to `tables/rerun/`; the checked-in
`tables/experiment_*.tex` files contain the reference numerical outputs.
The manuscript uses its own formatting of these two tables. The heavy-tail and
penalty scripts write CSV and LaTeX files directly to `tables/`. To synchronize
these two tables with the manuscript:

```bash
cp tables/heavytail_results.tex tables/grid_penalties.tex ../asymptotic_evalues_bentkus/tables/
```

Open each notebook in Jupyter and run all cells from this directory to reproduce
its calculations. Jupyter is only needed for the notebook interface and is not
included in the experiment dependencies. Saved cell outputs are omitted; the
manuscript figures are included in `plots/`.

## Experimental settings

The Gaussian comparison uses seed 42, 100 hypotheses, and 100 simulations per
scenario. Nominal non-null proportions 0.01, 0.025, 0.05, 0.075, and 0.1 give
actual non-null counts 1, 2, 5, 7, and 10 through `int(p * K)`. All methods use
the same draws. The table reports total rejections.

The heavy-tail experiment uses `X = mu + t_3 / sqrt(3)` and the statistic
`sum(X) / sqrt(sum(X**2))`, without centering the denominator. Five of 100
hypotheses have `mu = 3.5 / sqrt(n)`; the others have zero mean. The e-BH target
FDR is 0.1. Each sample size `n = 50, 200, 1000` uses 1,000 repetitions and
seed `1234 + n`. The CSV reports total, true, and false discoveries, empirical
FDR, and the Monte Carlo standard error of each mean. Standard errors use the
sample standard deviation (`ddof=1`) divided by the square root of the repetition
count. FDR is the mean of `false / max(rejections, 1)`.

The heavy-tail script defaults to batches of 50; use `--batch-size 10` to reduce
memory. `--repetitions` and `--output` are also available. Keep 1,000 repetitions
to reproduce the manuscript; use a separate output path for exploratory runs:

```bash
python3 heavytail_experiment.py --repetitions 10 --output /tmp/heavytail_smoke.csv
```

Finite-grid penalty decimals are numerical estimates, not certified lower
bounds. The analytic guarantee is `1/m`. For polynomial families the script
evaluates envelope crossings and stationary points, including the limit at
infinity. For the exponential family that limit is `1/m`.

At `Z = 3`, the alpha=1 post-hoc reciprocal is approximately 0.0100015. Its
rounded table entry of 0.010 does not imply rejection at exactly 0.01.
