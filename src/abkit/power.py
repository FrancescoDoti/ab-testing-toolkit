"""Sample size per arm for a two-sided test, using the normal approximation."""
import math
from scipy.stats import norm


def sample_size_means(sigma: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Users per arm needed to detect an absolute difference `mde` in a mean with std `sigma`."""
    if sigma <= 0 or mde == 0:
        raise ValueError("sigma must be > 0 and mde must be non-zero")
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return math.ceil(2 * (z * sigma / mde) ** 2)


def sample_size_proportions(p_base: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Users per arm to detect p_base -> p_base + mde_abs (pooled variance under H0, unpooled under H1)."""
    p1, p2 = p_base, p_base + mde_abs
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("proportions must lie strictly between 0 and 1")
    pbar = (p1 + p2) / 2
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    num = za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil((num / mde_abs) ** 2)
