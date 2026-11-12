"""CUPED (Deng et al., 2013): reduce variance using a pre-experiment covariate.

Y_adj = Y - theta * (X - mean(X)), with theta = cov(X, Y) / var(X) estimated on
both arms pooled. Because X is measured before randomization it is independent of
the assignment, so the adjustment leaves the treatment effect unbiased while
cutting variance by a factor of (1 - corr(X, Y)^2).
"""
import numpy as np


def cuped_theta(y: np.ndarray, x: np.ndarray) -> float:
    y, x = np.asarray(y, float), np.asarray(x, float)
    vx = x.var(ddof=1)
    return 0.0 if vx == 0 else float(np.cov(x, y, ddof=1)[0, 1] / vx)


def cuped_adjust(y_c, x_c, y_t, x_t) -> tuple[np.ndarray, np.ndarray, float]:
    """Return adjusted control and treatment outcomes plus the pooled theta."""
    y_c, x_c, y_t, x_t = (np.asarray(a, float) for a in (y_c, x_c, y_t, x_t))
    x_all = np.concatenate([x_c, x_t])
    theta = cuped_theta(np.concatenate([y_c, y_t]), x_all)
    mu = x_all.mean()
    return y_c - theta * (x_c - mu), y_t - theta * (x_t - mu), theta
