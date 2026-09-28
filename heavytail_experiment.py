#!/usr/bin/env python3
"""Reproduce the manuscript's heavy-tailed raw-sample e-BH experiment.

Setup:
- K=100 hypotheses, 5 non-nulls
- target FDR = 0.1
- X_{k,i} = mu_{k,n} + eps_{k,i}
- eps ~ t_3 / sqrt(3), so Var(eps)=1 and E|eps|^3=infinity
- mu_{k,n}=3.5/sqrt(n) for non-nulls
- 10-point uniform mixtures tuned to delta in [0.001, 0.02]
- 1,000 Monte Carlo repetitions
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm


K = 100
FDR_LEVEL = 0.1
NONNULL_COUNT = 5
SIGNAL = 3.5
ALPHAS = (1, 2, math.inf)


def gaussian_survival(x: np.ndarray | float) -> np.ndarray | float:
    return norm.sf(x)


def truncated_moment(alpha: int, x: np.ndarray | float) -> np.ndarray | float:
    """I_alpha(x) = E[(Z-x)_+^alpha] for alpha in {1,2}."""
    if alpha == 1:
        return norm.pdf(x) - x * gaussian_survival(x)
    if alpha == 2:
        return (1.0 + np.asarray(x) ** 2) * gaussian_survival(x) - x * norm.pdf(x)
    raise ValueError("This script supports alpha=1 or alpha=2.")


def threshold(lam: float, delta: float, alpha: int) -> float:
    return float(lam + (truncated_moment(alpha, lam) / delta) ** (1.0 / alpha))


def optimal_lambda(delta: float, alpha: int | float) -> float:
    if math.isinf(alpha):
        return math.sqrt(2.0 * math.log(1.0 / delta))
    result = minimize_scalar(
        lambda lam: threshold(lam, delta, int(alpha)),
        bounds=(-5.0, 8.0),
        method="bounded",
    )
    if not result.success:
        raise RuntimeError(f"Lambda optimization failed: {result.message}")
    return float(result.x)


def make_lambda_grids() -> dict[int | float, np.ndarray]:
    delta_min = FDR_LEVEL / K
    delta_max = 0.2 * FDR_LEVEL
    grids: dict[int | float, np.ndarray] = {}
    for alpha in ALPHAS:
        lo = optimal_lambda(delta_max, alpha)
        hi = optimal_lambda(delta_min, alpha)
        grids[alpha] = np.linspace(lo, hi, 10)
    return grids


def mixture_evalues(
    z: np.ndarray,
    alpha: int | float,
    lambdas: np.ndarray,
) -> np.ndarray:
    if math.isinf(alpha):
        component_values = np.exp(
            z[..., None] * lambdas - 0.5 * lambdas**2
        )
    else:
        a = int(alpha)
        component_values = (
            np.maximum(z[..., None] - lambdas, 0.0) ** a
            / truncated_moment(a, lambdas)
        )
    return component_values.mean(axis=-1)


def ebh_metrics(
    evalues: np.ndarray,
    nonnull_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return total, true, false discoveries and FDP for each replication."""
    batch_size, hypothesis_count = evalues.shape
    order = np.argsort(-evalues, axis=1)
    sorted_evalues = np.take_along_axis(evalues, order, axis=1)

    ranks = np.arange(1, hypothesis_count + 1)
    passes = sorted_evalues >= hypothesis_count / (ranks * FDR_LEVEL)
    total = np.where(passes, ranks, 0).max(axis=1)

    repeated_mask = np.broadcast_to(nonnull_mask, (batch_size, hypothesis_count))
    sorted_nonnull = np.take_along_axis(repeated_mask, order, axis=1)
    cumulative_true = np.cumsum(sorted_nonnull, axis=1)
    true = np.where(
        total > 0,
        cumulative_true[np.arange(batch_size), np.maximum(total - 1, 0)],
        0,
    )
    false = total - true
    fdp = false / np.maximum(total, 1)
    return total, true, false, fdp


def simulate_one_n(
    n: int,
    repetitions: int,
    batch_size: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(1234 + n)
    nonnull_mask = np.zeros(K, dtype=bool)
    nonnull_mask[:NONNULL_COUNT] = True
    lambda_grids = make_lambda_grids()

    metrics = {
        alpha: {"rejections": [], "true": [], "false": [], "fdp": []}
        for alpha in ALPHAS
    }

    for start in range(0, repetitions, batch_size):
        current_batch = min(batch_size, repetitions - start)
        noise = rng.standard_t(3, size=(current_batch, K, n)) / math.sqrt(3.0)
        mean_shift = nonnull_mask[None, :, None] * (SIGNAL / math.sqrt(n))
        observations = noise + mean_shift

        z = observations.sum(axis=2) / np.sqrt(
            np.square(observations).sum(axis=2)
        )

        for alpha in ALPHAS:
            evalues = mixture_evalues(z, alpha, lambda_grids[alpha])
            total, true, false, fdp = ebh_metrics(evalues, nonnull_mask)
            metrics[alpha]["rejections"].append(total)
            metrics[alpha]["true"].append(true)
            metrics[alpha]["false"].append(false)
            metrics[alpha]["fdp"].append(fdp)

    rows: list[dict[str, float | int | str]] = []
    for alpha in ALPHAS:
        arrays = {
            name: np.concatenate(parts)
            for name, parts in metrics[alpha].items()
        }
        label = "exponential" if math.isinf(alpha) else f"alpha={int(alpha)}"
        rows.append(
            {
                "n": n,
                "family": label,
                "mean_rejections": arrays["rejections"].mean(),
                "se_rejections": arrays["rejections"].std(ddof=1)
                / math.sqrt(repetitions),
                "mean_true_discoveries": arrays["true"].mean(),
                "se_true_discoveries": arrays["true"].std(ddof=1)
                / math.sqrt(repetitions),
                "mean_false_discoveries": arrays["false"].mean(),
                "se_false_discoveries": arrays["false"].std(ddof=1)
                / math.sqrt(repetitions),
                "empirical_fdr": arrays["fdp"].mean(),
                "se_fdr": arrays["fdp"].std(ddof=1) / math.sqrt(repetitions),
            }
        )
    return pd.DataFrame(rows)


def write_latex_table(result: pd.DataFrame, path: Path) -> None:
    """Format the manuscript table with booktabs and discovery-count colors.

    Per-row Monte Carlo standard errors remain in the CSV; their bounds are
    reported in the paper text alongside the table.
    """
    lines = [r"\begin{tabular}{rlrrrr}", r"\toprule",
             r"$n$ & Family & Rejections & True & False & FDR \\", r"\midrule"]
    for n, group in result.groupby("n", sort=False):
        if n != result.iloc[0]["n"]:
            lines.append(r"\midrule")
        for row in group.itertuples():
            label = {"alpha=1": r"$\alpha = 1$", "alpha=2": r"$\alpha = 2$",
                     "exponential": r"$\alpha = \infty$"}[row.family]
            values = []
            for metric in ["mean_rejections", "mean_true_discoveries",
                           "mean_false_discoveries", "empirical_fdr"]:
                value = getattr(row, metric)
                cell = f"{value:.3f}"
                if metric in ["mean_rejections", "mean_true_discoveries"]:
                    if value == group[metric].max():
                        cell = r"{\color{blue}" + cell + "}"
                    elif value == group[metric].min():
                        cell = r"{\color{red}" + cell + "}"
                values.append(cell)
            lines.append(f"{row.n} & {label} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "tables" / "heavytail_results.csv",
    )
    args = parser.parse_args()
    if args.repetitions < 2 or args.batch_size < 1:
        parser.error("--repetitions must be at least 2 and --batch-size positive")

    result = pd.concat(
        [
            simulate_one_n(n, args.repetitions, args.batch_size)
            for n in (50, 200, 1000)
        ],
        ignore_index=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    write_latex_table(result, args.output.with_suffix(".tex"))
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
