"""Monte Carlo study: does each method deliver the error rates it promises?

Run: python -m abkit.study [--sims N] [--seed S]
Writes results/report.md, results/results.json and figures to results/.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from .cuped import cuped_adjust
from .power import sample_size_means
from .sequential import msprt_always_valid_pvalues
from .srm import srm_check
from .tests import ratio_metric_test, welch_ttest


def peeking_study(rng, sims: int, n_per_arm: int = 5000, looks: int = 20, effect: float = 0.0, sigma: float = 1.0, alpha: float = 0.05):
    """Simulate cumulative means at evenly spaced looks; compare three stopping rules."""
    step = n_per_arm // looks
    ns = step * np.arange(1, looks + 1)
    # Per-look batch means are sufficient statistics: draw batch sums directly.
    c = rng.normal(0, sigma * np.sqrt(step), (sims, looks)).cumsum(1) / ns
    t = rng.normal(effect * step, sigma * np.sqrt(step), (sims, looks)).cumsum(1) / ns
    d = t - c
    se = sigma * np.sqrt(2 / ns)  # known-variance z-test keeps the comparison clean
    p_naive = 2 * stats.norm.sf(np.abs(d / se))
    naive_any = (p_naive <= alpha).any(1)
    fixed = p_naive[:, -1] <= alpha
    tau2 = (0.05 * sigma) ** 2
    stops = []
    for i in range(sims):
        r = msprt_always_valid_pvalues(d[i], se**2, tau2, alpha)
        stops.append(r.rejected_at)
    msprt_rej = np.array([s is not None for s in stops])
    stop_n = np.array([ns[s] if s is not None else ns[-1] for s in stops])
    return {
        "effect": effect, "looks": looks, "n_per_arm": n_per_arm, "sims": sims,
        "fixed_horizon_reject_rate": float(fixed.mean()),
        "naive_peeking_reject_rate": float(naive_any.mean()),
        "msprt_reject_rate": float(msprt_rej.mean()),
        "msprt_mean_sample_when_rejecting": float(stop_n[msprt_rej].mean()) if msprt_rej.any() else None,
    }


def cuped_study(rng, sims: int, n: int = 2000, rhos=(0.0, 0.3, 0.5, 0.7, 0.9), effect: float = 0.05):
    rows = []
    for rho in rhos:
        raw_est, adj_est, cover = [], [], 0
        for _ in range(sims):
            x = rng.normal(size=2 * n)
            y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=2 * n)
            y[n:] += effect
            yc, yt, _ = cuped_adjust(y[:n], x[:n], y[n:], x[n:])
            raw_est.append(y[n:].mean() - y[:n].mean())
            r = welch_ttest(yc, yt)
            adj_est.append(r.effect)
            cover += r.ci_low <= effect <= r.ci_high
        raw_v, adj_v = np.var(raw_est), np.var(adj_est)
        rows.append({
            "rho": rho,
            "variance_reduction_observed": float(1 - adj_v / raw_v),
            "variance_reduction_theory": float(rho**2),
            "adjusted_bias": float(np.mean(adj_est) - effect),
            "ci95_coverage": cover / sims,
        })
    return rows


def ratio_study(rng, sims: int, users: int = 2000):
    """Clicks/sessions CTR under H0 with per-user heterogeneity. Naive = treat sessions as iid."""
    cover_delta = cover_naive = 0
    for _ in range(sims):
        arms = []
        for _ in range(2):
            sessions = rng.poisson(5, users) + 1
            user_ctr = rng.beta(2, 8, users)  # users differ: clicks within a user are correlated
            clicks = rng.binomial(sessions, user_ctr)
            arms.append((clicks, sessions))
        (kc, sc), (kt, st) = arms
        r = ratio_metric_test(kc, sc, kt, st)
        cover_delta += r.ci_low <= 0 <= r.ci_high
        pc, pt = kc.sum() / sc.sum(), kt.sum() / st.sum()
        se = np.sqrt(pc * (1 - pc) / sc.sum() + pt * (1 - pt) / st.sum())
        cover_naive += abs(pt - pc) <= 1.96 * se
    return {"sims": sims, "delta_method_ci95_coverage": cover_delta / sims, "naive_session_level_ci95_coverage": cover_naive / sims}


def power_study(rng, sims: int, sigma: float = 1.0, mde: float = 0.1):
    n = sample_size_means(sigma, mde)
    rej = 0
    for _ in range(sims):
        c = rng.normal(0, sigma, n)
        t = rng.normal(mde, sigma, n)
        rej += welch_ttest(c, t).p_value <= 0.05
    return {"n_per_arm": n, "target_power": 0.8, "empirical_power": rej / sims, "sims": sims}


def srm_study(rng, sims: int, users: int = 100_000):
    alarms_ok = alarms_bug = 0
    for _ in range(sims):
        a = rng.binomial(users, 0.5)
        alarms_ok += srm_check([a, users - a], [1, 1])[1]
        b = rng.binomial(users, 0.495)  # a 1% relative assignment bug
        alarms_bug += srm_check([b, users - b], [1, 1])[1]
    return {"false_alarm_rate": alarms_ok / sims, "detection_rate_for_49.5_50.5_split": alarms_bug / sims, "threshold": 0.001, "sims": sims}


def plot(results: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    null = results["peeking_null"]
    labels = ["Fixed horizon\n(1 look)", "Naive peeking\n(20 looks)", "mSPRT\n(20 looks)"]
    vals = [null["fixed_horizon_reject_rate"], null["naive_peeking_reject_rate"], null["msprt_reject_rate"]]
    ax[0].bar(labels, vals, color=["#6b7280", "#dc2626", "#2563eb"])
    ax[0].axhline(0.05, ls="--", c="k", lw=1)
    ax[0].set_title("False positive rate under H0 (alpha = 0.05)")
    for i, v in enumerate(vals):
        ax[0].text(i, v + 0.003, f"{v:.3f}", ha="center")
    rows = results["cuped"]
    r = [x["rho"] for x in rows]
    ax[1].plot(r, [x["variance_reduction_theory"] for x in rows], "k--", label="theory: rho^2")
    ax[1].plot(r, [x["variance_reduction_observed"] for x in rows], "o-", c="#2563eb", label="observed")
    ax[1].set_xlabel("corr(pre-period metric, outcome)")
    ax[1].set_ylabel("variance reduction")
    ax[1].set_title("CUPED variance reduction")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "study.png", dpi=140)


def report(res: dict) -> str:
    n, a = res["peeking_null"], res["peeking_effect"]
    lines = [
        "# Simulation study results", "",
        f"Seed {res['seed']}. Every number below is produced by `python -m abkit.study`; rerun to reproduce.", "",
        "## 1. Peeking: fixed horizon vs naive repeated testing vs mSPRT", "",
        f"{n['sims']} simulated experiments, {n['n_per_arm']} users per arm, {n['looks']} interim looks.", "",
        "| Rule | Reject rate, no true effect | Reject rate, effect = 0.05 sd |",
        "|---|---|---|",
        f"| Fixed horizon (analyse once at the end) | {n['fixed_horizon_reject_rate']:.3f} | {a['fixed_horizon_reject_rate']:.3f} |",
        f"| Naive peeking (stop at first p < 0.05) | **{n['naive_peeking_reject_rate']:.3f}** | {a['naive_peeking_reject_rate']:.3f} |",
        f"| mSPRT always-valid p-value | {n['msprt_reject_rate']:.3f} | {a['msprt_reject_rate']:.3f} |", "",
        f"When mSPRT rejects under the true effect it does so after {a['msprt_mean_sample_when_rejecting']:.0f} users per arm on average (of {a['n_per_arm']}).", "",
        "## 2. CUPED", "",
        "| corr | variance reduction (observed) | theory (rho^2) | bias | 95% CI coverage |", "|---|---|---|---|---|",
    ]
    for r in res["cuped"]:
        lines.append(f"| {r['rho']} | {r['variance_reduction_observed']:.3f} | {r['variance_reduction_theory']:.3f} | {r['adjusted_bias']:+.4f} | {r['ci95_coverage']:.3f} |")
    d = res["ratio"]
    p = res["power"]
    s = res["srm"]
    lines += [
        "", "## 3. Ratio metrics (CTR = clicks / sessions), no true effect", "",
        "| Standard error | 95% CI coverage |", "|---|---|",
        f"| Delta method, user as the unit | {d['delta_method_ci95_coverage']:.3f} |",
        f"| Naive, sessions treated as independent | **{d['naive_session_level_ci95_coverage']:.3f}** |", "",
        "## 4. Power calculator", "",
        f"Sample size for MDE = 0.1 sd at 80% power: {p['n_per_arm']} per arm. Empirical power at that size: {p['empirical_power']:.3f}.", "",
        "## 5. Sample ratio mismatch check (p < 0.001)", "",
        f"False alarm rate with a correct 50/50 split: {s['false_alarm_rate']:.4f}. Detection rate for a 49.5/50.5 split at 100k users: {s['detection_rate_for_49.5_50.5_split']:.3f}.", "",
        "![study](study.png)", "",
    ]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = args.sims
    res = {
        "seed": args.seed,
        "peeking_null": peeking_study(rng, s * 5),
        "peeking_effect": peeking_study(rng, s, effect=0.05),
        "cuped": cuped_study(rng, s),
        "ratio": ratio_study(rng, s // 2),
        "power": power_study(rng, s),
        "srm": srm_study(rng, s * 5),
    }
    (out / "results.json").write_text(json.dumps(res, indent=2))
    (out / "report.md").write_text(report(res))
    plot(res, out)
    print(report(res))


if __name__ == "__main__":
    main()
