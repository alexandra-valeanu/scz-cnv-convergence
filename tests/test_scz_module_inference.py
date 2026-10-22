import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from scz_module_inference import module_counts, module_statistics, module_tests


def test_distinct_genes_in_same_module_converge_but_one_cnv_does_not():
    membership = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)
    reach = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=bool)
    counts = module_counts(reach, membership)
    np.testing.assert_equal(counts, [[1, 0], [1, 0]])
    stats = module_statistics(counts, reach.sum(axis=1), [2, 2])
    np.testing.assert_allclose(stats['pair_coverage'], [.25, 0])
    one = module_statistics(np.array([[2, 0], [0, 0]]), [2, 0], [2, 2])
    np.testing.assert_equal(one['pair_coverage'], [0, 0])
    assert np.isfinite(one['landscape_share']).all()


def test_empirical_tails_and_families_include_zero_hits():
    observed = np.array([[1, 0], [1, 0]])
    null = np.array([[[0, 0], [0, 0]], [[1, 0], [0, 0]],
                     [[1, 0], [1, 0]], [[2, 0], [2, 0]]])
    result = module_tests(observed, null, [2, 1], null.sum(axis=-1), [2, 2], ['A', 'B'], ['x', 'y'], 'direct')
    convergence = result.query("endpoint == 'pair_coverage'").set_index('module')
    assert convergence.loc['A', 'p'] == pytest.approx(3/5)
    assert convergence.loc['B', 'p'] == 1
    assert convergence.family_size.tolist() == [2, 2]
    assert convergence.p_holm.tolist() == [1, 1]
    cells = result.query("endpoint == 'coverage'")
    assert cells.family_size.tolist() == [4]*4
    shares = result.query("endpoint == 'landscape_share' and module == 'A'")
    assert shares.observed.tolist() == [.5, 1.]


def test_bad_module_shapes_rejected():
    with pytest.raises(ValueError, match='aligned'):
        module_tests(np.ones((2, 2)), np.ones((3, 2, 2)), [2, 2], np.ones((3, 2)),
                     [0, 2], ['a', 'b'], ['x', 'y'], 'direct')


def test_leave_one_out_removes_the_only_second_contributor():
    from scz_module_report import leave_one_out
    data = {'cnvs': np.array(['a', 'b', 'c']), 'modules': np.array(['M']),
            'module_sizes': np.array([2]), 'observed_counts': np.array([[1], [1], [0]]),
            'observed_totals': np.array([1, 1, 0]), 'counts': np.zeros((4, 3, 1)),
            'totals': np.zeros((4, 3))}
    result = leave_one_out(data).set_index('omitted_cnv')
    assert result.loc['a', 'observed'] == 0
    assert result.loc['b', 'observed'] == 0
    assert result.loc['c', 'observed'] == .25
    assert result.loc['a', 'p'] == 1
    assert result.loc['c', 'p'] == .2
