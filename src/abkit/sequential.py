"""Always-valid inference with the mixture SPRT (Johari, Pekelis & Walsh, 2017).

A fixed-horizon test checked every day ("peeking") inflates the false positive
rate far above alpha. The mSPRT's likelihood ratio, mixed over a N(0, tau^2) prior
on the effect, is a nonnegative martingale under H0, so by Ville's inequality
P(sup_n Lambda_n >= 1/alpha) <= alpha: you may stop the first time it crosses.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MSPRTResult:
    p_values: np.ndarray        # always-valid p-value after each look
    rejected_at: int | None     # index of the first look with p <= alpha, if any


def msprt_always_valid_pvalues(diffs: np.ndarray, variances: np.ndarray, tau2: float, alpha: float = 0.05) -> MSPRTResult:
    """diffs[k]: estimated treatment-control difference at look k; variances[k]: its variance.

    Lambda = sqrt(V / (V + tau2)) * exp(tau2 * d^2 / (2 V (V + tau2))), tested against theta0 = 0.
    tau2 should be on the scale of plausible effects squared (e.g. the MDE^2).
    """
    d = np.asarray(diffs, float)
    v = np.asarray(variances, float)
    if tau2 <= 0 or np.any(v <= 0):
        raise ValueError("tau2 and all variances must be positive")
    log_lam = 0.5 * np.log(v / (v + tau2)) + tau2 * d**2 / (2 * v * (v + tau2))
    p = np.minimum.accumulate(np.minimum(1.0, np.exp(-log_lam)))
    hits = np.flatnonzero(p <= alpha)
    return MSPRTResult(p, int(hits[0]) if hits.size else None)
