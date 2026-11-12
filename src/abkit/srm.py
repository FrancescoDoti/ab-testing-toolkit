"""Sample ratio mismatch: is the observed split consistent with the intended one?"""
from scipy.stats import chisquare


def srm_check(counts: list[int], expected_ratio: list[float], threshold: float = 0.001) -> tuple[float, bool]:
    """Chi-square goodness-of-fit. Returns (p_value, mismatch). A tiny threshold is the
    industry convention because SRM checks run on every experiment and a real mismatch
    usually means broken assignment or logging, not bad luck."""
    total = sum(counts)
    s = sum(expected_ratio)
    expected = [total * r / s for r in expected_ratio]
    p = float(chisquare(counts, expected).pvalue)
    return p, p < threshold
