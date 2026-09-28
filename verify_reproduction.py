#!/usr/bin/env python3
"""Check saved reproduction outputs against the manuscript tables.

Run the heavy-tail and grid-penalty scripts first. This check does not claim that a saved
CSV substitutes for rerunning the simulations.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from heavytail_experiment import ebh_metrics, write_latex_table


TABLES = Path(__file__).resolve().parent / 'tables'
EXPECTED = np.array([
    [3.306, 3.256, .050, .010], [3.231, 3.184, .047, .010],
    [2.644, 2.633, .011, .003], [3.360, 3.295, .065, .014],
    [3.295, 3.237, .058, .013], [2.842, 2.815, .027, .006],
    [3.432, 3.328, .104, .025], [3.368, 3.277, .091, .022],
    [2.846, 2.808, .038, .011],
])


def main():
    heavy = pd.read_csv(TABLES / 'heavytail_results.csv')
    assert list(heavy.n) == [50] * 3 + [200] * 3 + [1000] * 3
    assert list(heavy.family) == ['alpha=1', 'alpha=2', 'exponential'] * 3
    cols = ['mean_rejections', 'mean_true_discoveries',
            'mean_false_discoveries', 'empirical_fdr']
    np.testing.assert_array_equal(heavy[cols].round(3).to_numpy(), EXPECTED)
    assert heavy.se_rejections.max() <= .043
    assert heavy.se_fdr.max() <= .0026
    for df, fdr in [(heavy, 'empirical_fdr')]:
        np.testing.assert_allclose(df.mean_rejections,
                                   df.mean_true_discoveries + df.mean_false_discoveries)
        assert df[fdr].between(0, 1).all()
        assert (df.filter(regex='^se_') >= 0).all().all()
    assert (heavy.mean_true_discoveries <= 5).all()

    # Check e-BH against rejection sets defined directly by its self-consistency
    # cutoff; include a step-up case where rank 1 fails but rank 2 passes.
    values = np.array([[18., 18.], [0., 0.], [20., 0.], [30., 20.]])
    truth = np.array([True, False])
    observed = ebh_metrics(values, truth)
    expected = [[], [], [], []]
    for row in values:
        k = max([r for r in range(1, 3) if (row >= 2 / (.1 * r)).sum() >= r], default=0)
        rejected = row >= 2 / (.1 * k) if k else np.zeros(2, dtype=bool)
        t = (rejected & truth).sum()
        v = (rejected & ~truth).sum()
        for bucket, value in zip(expected, [k, t, v, v / max(k, 1)]):
            bucket.append(value)
    np.testing.assert_allclose(observed, expected)

    # Regenerate LaTeX from the full-precision outputs and compare exactly.
    with TemporaryDirectory() as directory:
        jobs = [(heavy, write_latex_table, 'heavytail_results.tex')]
        for df, writer, name in jobs:
            generated = Path(directory) / name
            writer(df, generated)
            assert generated.read_text() == (TABLES / name).read_text(), name
            paper = TABLES.parent.parent / 'asymptotic_evalues_bentkus' / 'tables' / name
            if paper.exists():
                assert generated.read_text() == paper.read_text(), name
    penalties = pd.read_csv(TABLES / 'grid_penalties.csv')
    assert (penalties.kappa >= penalties.universal - 1e-10).all()
    assert (penalties.kappa <= 1).all()
    print('PASS: all 36 heavy-tail entries, MCSE bounds, discovery accounting, e-BH edge cases, and manuscript heavy-tail table.')


if __name__ == '__main__':
    main()
