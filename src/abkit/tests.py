"""Fixed-horizon tests: Welch t-test, two-proportion z-test, ratio metrics via the delta method."""
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    effect: float          # treatment - control (absolute)
    std_error: float
    p_value: float
    ci_low: float
    ci_high: float


def _result(effect: float, se: float, alpha: float, df: float | None = None) -> TestResult:
    dist = stats.t(df) if df is not None else stats.norm
    stat = effect / se if se > 0 else 0.0
    p = 2 * dist.sf(abs(stat))
    q = dist.ppf(1 - alpha / 2)
    return TestResult(effect, se, float(p), effect - q * se, effect + q * se)


def welch_ttest(control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05) -> TestResult:
    c, t = np.asarray(control, float), np.asarray(treatment, float)
    vc, vt = c.var(ddof=1) / len(c), t.var(ddof=1) / len(t)
    se = np.sqrt(vc + vt)
    df = (vc + vt) ** 2 / (vc**2 / (len(c) - 1) + vt**2 / (len(t) - 1))
    return _result(t.mean() - c.mean(), se, alpha, df)


def two_proportion_ztest(conv_c: int, n_c: int, conv_t: int, n_t: int, alpha: float = 0.05) -> TestResult:
    """p-value uses the pooled SE (H0), the confidence interval the unpooled SE."""
    pc, pt = conv_c / n_c, conv_t / n_t
    pool = (conv_c + conv_t) / (n_c + n_t)
    se0 = np.sqrt(pool * (1 - pool) * (1 / n_c + 1 / n_t))
    se1 = np.sqrt(pc * (1 - pc) / n_c + pt * (1 - pt) / n_t)
    effect = pt - pc
    p = 2 * stats.norm.sf(abs(effect / se0)) if se0 > 0 else 1.0
    q = stats.norm.ppf(1 - alpha / 2)
    return TestResult(effect, se1, float(p), effect - q * se1, effect + q * se1)


def _ratio_mean_var(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    """Delta-method variance of sum(num)/sum(den) when the randomization unit is the user.

    Clicks/sessions per user are correlated within a user, so treating sessions as
    independent understates the variance; the delta method keeps the user as the unit.
    """
    n = len(num)
    mx, my = num.mean(), den.mean()
    vx, vy = num.var(ddof=1), den.var(ddof=1)
    cxy = np.cov(num, den, ddof=1)[0, 1]
    r = mx / my
    var = (vx - 2 * r * cxy + r**2 * vy) / (my**2 * n)
    return r, var


def ratio_metric_test(num_c, den_c, num_t, den_t, alpha: float = 0.05) -> TestResult:
    rc, vc = _ratio_mean_var(np.asarray(num_c, float), np.asarray(den_c, float))
    rt, vt = _ratio_mean_var(np.asarray(num_t, float), np.asarray(den_t, float))
    return _result(rt - rc, float(np.sqrt(vc + vt)), alpha)
