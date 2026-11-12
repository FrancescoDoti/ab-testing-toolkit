import numpy as np
import pytest
from scipy import stats

from abkit import (cuped_adjust, msprt_always_valid_pvalues, ratio_metric_test, sample_size_means,
                   sample_size_proportions, srm_check, two_proportion_ztest, welch_ttest)


def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    c, t = rng.normal(0, 1, 300), rng.normal(0.2, 2, 500)
    ours = welch_ttest(c, t)
    ref = stats.ttest_ind(t, c, equal_var=False)
    assert ours.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert ours.ci_low < ours.effect < ours.ci_high


def test_two_proportion_known_case():
    r = two_proportion_ztest(100, 1000, 130, 1000)
    assert r.effect == pytest.approx(0.03)
    assert 0.03 < r.p_value < 0.05  # z ~ 2.10 with pooled SE


def test_sample_size_textbook_values():
    assert sample_size_means(1.0, 0.1) == 1570  # 2 * (1.96 + 0.8416)^2 / 0.01
    assert 3800 < sample_size_proportions(0.10, 0.02) < 3900
    with pytest.raises(ValueError):
        sample_size_means(0, 1)


def test_cuped_removes_variance_without_bias():
    rng = np.random.default_rng(1)
    n = 20000
    x = rng.normal(size=2 * n)
    y = 0.8 * x + 0.6 * rng.normal(size=2 * n)
    y[n:] += 0.1
    yc, yt, theta = cuped_adjust(y[:n], x[:n], y[n:], x[n:])
    assert theta == pytest.approx(0.8, abs=0.02)
    assert (yt.mean() - yc.mean()) == pytest.approx(0.1, abs=0.02)
    assert np.var(np.concatenate([yc, yt])) < 0.5 * np.var(y)


def test_ratio_metric_wider_than_naive_with_clustering():
    rng = np.random.default_rng(2)
    s = rng.poisson(5, 3000) + 1
    k = rng.binomial(s, rng.beta(2, 8, 3000))
    r = ratio_metric_test(k[:1500], s[:1500], k[1500:], s[1500:])
    p = k.sum() / s.sum()
    naive_se = np.sqrt(2 * p * (1 - p) / s[:1500].sum())
    assert r.std_error > naive_se


def test_srm():
    assert srm_check([50000, 50000], [1, 1])[1] is False
    assert srm_check([49000, 51000], [1, 1])[1] is True


def test_msprt_pvalues_monotone_and_reject_large_effect():
    n = np.arange(100, 10001, 100)
    v = 2 / n
    r = msprt_always_valid_pvalues(np.full(len(n), 0.2), v, tau2=0.01)
    assert np.all(np.diff(r.p_values) <= 0)
    assert r.rejected_at is not None
    null = msprt_always_valid_pvalues(np.zeros(len(n)), v, tau2=0.01)
    assert null.rejected_at is None
    with pytest.raises(ValueError):
        msprt_always_valid_pvalues([0.1], [0.0], tau2=0.01)


def test_msprt_controls_type_one_error_under_continuous_monitoring():
    rng = np.random.default_rng(3)
    sims, looks, step = 2000, 50, 100
    ns = step * np.arange(1, looks + 1)
    d = rng.normal(0, np.sqrt(2 * step), (sims, looks)).cumsum(1) / ns
    rej = sum(msprt_always_valid_pvalues(d[i], 2 / ns, tau2=0.0025).rejected_at is not None for i in range(sims))
    assert rej / sims <= 0.05 + 0.01
