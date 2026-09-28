#!/usr/bin/env python3
"""Compute finite-grid mixture/envelope infima on the paper's active domains.

For alpha=1,2 enumerate polynomial envelope crossings and stationary points,
including the unbounded final interval. Decimal output is numerical, not a
certified lower bound; 1/m is the analytic universal lower bound.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from scipy.stats import norm
from mixtures_experiments import (find_optimal_lambd_bentkus,
                                 find_optimal_lambd_exponential, I_k, G,
                                 U_delta_alpha)


def real_roots(poly):
    return [float(r.real) for r in poly.roots() if abs(r.imag) < 1e-8]


def grid_penalty(alpha, grid, delta_max):
    m = len(grid)
    if math.isinf(alpha):
        # The unique largest-lambda exponential dominates in the upper tail.
        return 1 / m
    z0 = min(float(U_delta_alpha(lam, delta_max, alpha)) for lam in grid)
    if alpha == 0:
        # Compare quantiles directly to avoid sf(isf(delta)) roundoff at anchors.
        z0 = float(min(grid[grid >= norm.isf(delta_max)]))
        values = []
        for z in sorted(set([z0, *grid[grid >= z0]])):
            e = np.where(z >= grid, 1 / G(grid), 0)
            values.append(e.mean() / e.max())
        return min(values)
    polynomials = [Polynomial([-lam, 1]) ** alpha / I_k(alpha, lam)
                   for lam in grid]
    breaks = {z0, *grid[grid > z0]}
    for i, p in enumerate(polynomials):
        for j in range(i):
            for root in real_roots(p - polynomials[j]):
                if root >= max(z0, grid[i], grid[j]):
                    breaks.add(root)
    breaks = sorted(breaks)

    def ratio(z):
        e = np.maximum(z - grid, 0) ** alpha / I_k(alpha, grid)
        return e.mean() / e.max()

    values = [ratio(z) for z in breaks]
    coeffs = 1 / I_k(alpha, grid)
    values.append(coeffs.mean() / coeffs.max())  # limit at infinity
    for left, right in zip(breaks, breaks[1:] + [math.inf]):
        mid = (left + right) / 2 if math.isfinite(right) else left + 1
        active = [p if mid > lam else Polynomial([0])
                  for p, lam in zip(polynomials, grid)]
        numerator = sum(active, Polynomial([0])) / m
        denominator = max(active, key=lambda p: p(mid))
        stationary = numerator.deriv() * denominator - numerator * denominator.deriv()
        for root in real_roots(stationary):
            if left < root < right:
                values.append(ratio(root))
    result = min(values)
    if not 1 / m - 1e-10 <= result <= 1 + 1e-10:
        raise AssertionError("Mixture penalty outside analytic bounds")
    return result


def write_latex_table(result, path):
    """Match the original family-row, booktabs layout."""
    lines = [r'\begin{tabular}{lrr}', r'\toprule',
             r'Family & Post-hoc & Testing \\', r'\midrule']
    for alpha in [0, 1, 2, math.inf]:
        label = r'$\alpha = \infty$' if math.isinf(alpha) else fr'$\alpha = {alpha}$'
        rows = result[result.alpha == alpha].set_index('experiment')
        cells = [f"{rows.loc[name, 'kappa']:.4f}" for name in ['Post-hoc', 'Testing']]
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    path.write_text('\n'.join(lines) + '\n')


def main():
    rows = []
    for name, deltas, m in [('Post-hoc', [0.1, 0.05, 0.01], 3),
                            ('Testing', [0.02, 0.001], 10)]:
        for alpha in [0, 1, 2, math.inf]:
            optimal = lambda d: (norm.isf(d) if alpha == 0
                                else find_optimal_lambd_exponential(d)
                                if math.isinf(alpha)
                                else find_optimal_lambd_bentkus(d, alpha))
            anchors = [optimal(d) for d in deltas]
            grid = np.array(anchors) if m == 3 else np.linspace(*anchors, m)
            rows.append({'experiment': name, 'alpha': alpha, 'm': m,
                         'delta_max': max(deltas), 'universal': 1 / m,
                         'kappa': grid_penalty(alpha, grid, max(deltas))})
    result = pd.DataFrame(rows)
    output = Path(__file__).resolve().parent / 'tables'
    output.mkdir(exist_ok=True)
    result.to_csv(output / 'grid_penalties.csv', index=False)
    write_latex_table(result, output / 'grid_penalties.tex')
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
