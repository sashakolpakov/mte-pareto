"""Likelihood-based finite-support Omega tail diagnostics."""

from __future__ import annotations

import csv
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def l_omega(n: int) -> int:
    """Elias Omega code length for a positive integer."""
    if n <= 0:
        raise ValueError("Elias Omega code length is defined for positive integers")

    length = 1
    k = n
    while k > 1:
        bits = k.bit_length()
        length += bits - 1
        k = bits - 1
    return length


@dataclass(frozen=True)
class ModelFit:
    name: str
    probabilities: np.ndarray
    params: dict[str, float]
    kl: float
    log_likelihood: float


@dataclass(frozen=True)
class BootstrapCI:
    n_bootstrap: int
    seed: int
    level: float
    slope_ci: tuple[float, float]
    kl_ci: tuple[float, float]


@dataclass(frozen=True)
class AnalysisResult:
    label: str
    vals: np.ndarray
    counts: np.ndarray
    raw_vals: np.ndarray
    raw_counts: np.ndarray
    tail_drop: int
    models: list[ModelFit]
    bootstrap_ci: BootstrapCI | None
    prior_figure: Path
    fitted_figure: Path
    metrics_csv: Path


def read_size_csv(path: str | Path) -> list[int]:
    """Read a one-column size CSV written by write_size_csv."""
    sizes: list[int] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return sizes
        field = "size_bytes" if "size_bytes" in reader.fieldnames else reader.fieldnames[0]
        for row in reader:
            try:
                size = int(row[field])
            except (KeyError, TypeError, ValueError):
                continue
            if size > 0:
                sizes.append(size)
    return sizes


def write_size_csv(path: str | Path, sizes: Iterable[int]) -> Path:
    """Write positive byte sizes as a deterministic one-column CSV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["size_bytes"])
        writer.writeheader()
        for size in sorted(int(size) for size in sizes if int(size) > 0):
            writer.writerow({"size_bytes": size})
    return out_path


def histogram_from_sizes(sizes: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    omegas = [l_omega(int(size)) for size in sizes if int(size) > 0]
    counts = Counter(omegas)
    if not counts:
        raise ValueError("No positive sizes were available for analysis")

    vals = np.array(sorted(counts), dtype=int)
    freqs = np.array([counts[v] for v in vals], dtype=float)
    if vals.size < 2:
        raise ValueError("At least two observed code lengths are needed for model fitting")
    return vals, freqs


def select_tail_support(
    vals: np.ndarray,
    counts: np.ndarray,
    tail_drop: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop the smallest observed codelength bins before fitting."""
    if tail_drop < 0:
        raise ValueError("tail_drop must be nonnegative")
    if tail_drop >= vals.size - 1:
        raise ValueError(
            f"tail_drop={tail_drop} leaves fewer than two codelength bins "
            f"from an observed support of size {vals.size}"
        )
    if tail_drop == 0:
        return vals, counts
    return vals[tail_drop:], counts[tail_drop:]


def _probs_from_feature(feature: np.ndarray, theta: float) -> np.ndarray:
    log_weights = -theta * feature
    log_weights = log_weights - np.max(log_weights)
    weights = np.exp(log_weights)
    return weights / np.sum(weights)


def _fit_one_parameter_feature(
    feature: np.ndarray,
    counts: np.ndarray,
    theta_min: float = -100.0,
    theta_max: float = 100.0,
) -> float:
    """Fit q_theta(x) proportional to exp(-theta * feature(x)).

    The multinomial MLE is the exponential-family moment match
    E_theta[feature] = empirical mean(feature).
    """
    target = float(np.average(feature, weights=counts))
    f_min = float(np.min(feature))
    f_max = float(np.max(feature))
    span = max(1.0, abs(f_max - f_min))
    tol = 1e-12 * span

    if target <= f_min + tol:
        return theta_max
    if target >= f_max - tol:
        return theta_min

    def model_mean(theta: float) -> float:
        return float(np.dot(_probs_from_feature(feature, theta), feature))

    if target >= model_mean(theta_min):
        return theta_min
    if target <= model_mean(theta_max):
        return theta_max

    lo = max(theta_min, -1.0)
    while model_mean(lo) < target:
        next_lo = lo * 2.0 if lo < 0.0 else -1.0
        if next_lo <= theta_min:
            return theta_min
        lo = next_lo

    hi = min(theta_max, 1.0)
    while model_mean(hi) > target:
        next_hi = hi * 2.0 if hi > 0.0 else 1.0
        if next_hi >= theta_max:
            return theta_max
        hi = next_hi

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if model_mean(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _kl_divergence(empirical: np.ndarray, model: np.ndarray) -> float:
    mask = empirical > 0
    return float(np.sum(empirical[mask] * np.log(empirical[mask] / model[mask])))


def _fit_metrics(
    name: str,
    counts: np.ndarray,
    probabilities: np.ndarray,
    params: dict[str, float] | None = None,
) -> ModelFit:
    n_obs = float(np.sum(counts))
    empirical = counts / n_obs
    log_probabilities = np.log(np.maximum(probabilities, np.finfo(float).tiny))
    log_likelihood = float(np.dot(counts, log_probabilities))
    return ModelFit(
        name=name,
        probabilities=probabilities,
        params=params or {},
        kl=_kl_divergence(empirical, probabilities),
        log_likelihood=log_likelihood,
    )


def fit_omega_models(vals: np.ndarray, counts: np.ndarray) -> list[ModelFit]:
    """Fit the Omega-tail diagnostic family on finite codelength support."""
    support = vals.astype(float)

    pure_omega = _probs_from_feature(support, math.log(2.0))

    scaled_a = _fit_one_parameter_feature(support, counts)
    scaled_omega = _probs_from_feature(support, scaled_a)

    return [
        _fit_metrics("pure Omega", counts, pure_omega, {"a": math.log(2.0)}),
        _fit_metrics("scaled Omega MLE", counts, scaled_omega, {"a": scaled_a}),
    ]


def bootstrap_scaled_omega(
    vals: np.ndarray,
    counts: np.ndarray,
    n_bootstrap: int,
    seed: int,
    level: float = 0.95,
) -> BootstrapCI | None:
    if n_bootstrap <= 0:
        return None

    rng = np.random.default_rng(seed)
    n_obs = int(np.sum(counts))
    empirical = counts / n_obs
    alpha = (1.0 - level) / 2.0
    slope_samples = np.empty(n_bootstrap, dtype=float)
    kl_samples = np.empty(n_bootstrap, dtype=float)

    for idx in range(n_bootstrap):
        sample_counts = rng.multinomial(n_obs, empirical)
        slope = _fit_one_parameter_feature(vals, sample_counts.astype(float))
        q = _probs_from_feature(vals, slope)
        sample_p = sample_counts / n_obs
        slope_samples[idx] = slope
        kl_samples[idx] = _kl_divergence(sample_p, q)

    return BootstrapCI(
        n_bootstrap=n_bootstrap,
        seed=seed,
        level=level,
        slope_ci=(
            float(np.quantile(slope_samples, alpha)),
            float(np.quantile(slope_samples, 1.0 - alpha)),
        ),
        kl_ci=(
            float(np.quantile(kl_samples, alpha)),
            float(np.quantile(kl_samples, 1.0 - alpha)),
        ),
    )


def _format_params(params: dict[str, float]) -> str:
    if not params:
        return "-"
    return ", ".join(f"{key}={value:.6g}" for key, value in params.items())


def print_report(result: AnalysisResult) -> None:
    vals = result.vals
    n_obs = int(np.sum(result.counts))
    raw_n = int(np.sum(result.raw_counts))
    dropped_n = raw_n - n_obs
    print(f"Dataset: {result.label}")
    print(
        f"Raw support: n={raw_n}, "
        f"{result.raw_vals.size} distinct Omega codelengths, "
        f"min={int(result.raw_vals[0])}, max={int(result.raw_vals[-1])}"
    )
    print(
        f"Analysis support: n={n_obs}, "
        f"{vals.size} distinct Omega codelengths, "
        f"min={int(vals[0])}, max={int(vals[-1])}"
    )
    if result.tail_drop:
        retained = n_obs / raw_n
        print(
            f"Dropped {result.tail_drop} smallest codelength bins "
            f"({dropped_n} observations; retained fraction={retained:.6f})."
        )
    print()
    print("Omega-tail diagnostic on finite analysis support (natural logs):")
    print(f"{'model':<20} {'params':<18} {'KL':>10} {'logLik':>14}")
    for fit in result.models:
        print(
            f"{fit.name:<20} {_format_params(fit.params):<18} "
            f"{fit.kl:>10.6f} {fit.log_likelihood:>14.3f}"
        )

    if result.bootstrap_ci is not None:
        ci = result.bootstrap_ci
        pct = 100.0 * ci.level
        print()
        print(
            f"Scaled Omega bootstrap CIs ({pct:.1f}%, "
            f"B={ci.n_bootstrap}, seed={ci.seed}):"
        )
        print(f"  slope a: [{ci.slope_ci[0]:.6g}, {ci.slope_ci[1]:.6g}]")
        print(f"  KL:      [{ci.kl_ci[0]:.6g}, {ci.kl_ci[1]:.6g}]")

    print()
    print(f"Saved metrics: {result.metrics_csv}")
    print(f"Saved figure:  {result.prior_figure}")
    print(f"Saved figure:  {result.fitted_figure}")


def write_metrics_csv(
    path: Path,
    models: list[ModelFit],
    n_obs: int,
    n_support: int,
    min_omega: int,
    max_omega: int,
    tail_drop: int,
    dropped_n: int,
    bootstrap_ci: BootstrapCI | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "parameters",
                "n_obs",
                "n_support",
                "min_omega",
                "max_omega",
                "tail_drop",
                "dropped_n",
                "kl",
                "log_likelihood",
                "bootstrap_level",
                "slope_ci_low",
                "slope_ci_high",
                "kl_ci_low",
                "kl_ci_high",
            ],
        )
        writer.writeheader()
        for fit in models:
            is_scaled = fit.name == "scaled Omega MLE" and bootstrap_ci is not None
            writer.writerow(
                {
                    "model": fit.name,
                    "parameters": _format_params(fit.params),
                    "n_obs": n_obs,
                    "n_support": n_support,
                    "min_omega": min_omega,
                    "max_omega": max_omega,
                    "tail_drop": tail_drop,
                    "dropped_n": dropped_n,
                    "kl": f"{fit.kl:.12g}",
                    "log_likelihood": f"{fit.log_likelihood:.12g}",
                    "bootstrap_level": f"{bootstrap_ci.level:.12g}" if is_scaled else "",
                    "slope_ci_low": f"{bootstrap_ci.slope_ci[0]:.12g}" if is_scaled else "",
                    "slope_ci_high": f"{bootstrap_ci.slope_ci[1]:.12g}" if is_scaled else "",
                    "kl_ci_low": f"{bootstrap_ci.kl_ci[0]:.12g}" if is_scaled else "",
                    "kl_ci_high": f"{bootstrap_ci.kl_ci[1]:.12g}" if is_scaled else "",
                }
            )


def _model_by_name(models: list[ModelFit], name: str) -> ModelFit:
    for fit in models:
        if fit.name == name:
            return fit
    raise KeyError(name)


def save_figures(
    label: str,
    vals: np.ndarray,
    counts: np.ndarray,
    models: list[ModelFit],
    output_prefix: str | Path,
    show: bool = False,
) -> tuple[Path, Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    mpl_config = prefix.parent / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    import matplotlib

    if not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    prior_path = Path(f"{prefix}_sizes_priors.png")
    fitted_path = Path(f"{prefix}_sizes_prior_scaled.png")

    empirical = counts / np.sum(counts)
    pure = _model_by_name(models, "pure Omega")
    scaled = _model_by_name(models, "scaled Omega MLE")

    plt.rcParams.update(
        {
            "figure.figsize": (6.4, 4.2),
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": ":",
        }
    )
    metadata = {"Software": "mte-pareto omega likelihood analysis"}

    fig, ax = plt.subplots(constrained_layout=True)
    ax.semilogy(vals, empirical, "o", color="black", label="empirical")
    ax.semilogy(vals, pure.probabilities, "-", color="#1f77b4", label="pure Omega, a=log 2")
    ax.set_xlabel(r"Elias Omega codelength $\ell_\Omega$")
    ax.set_ylabel(r"Probability mass on observed support")
    ax.set_title(f"{label}: pure Omega reference")
    ax.legend(frameon=False)
    fig.savefig(prior_path, bbox_inches="tight", metadata=metadata)
    if show:
        plt.show()
    plt.close(fig)

    scaled_a = scaled.params["a"]
    fig, ax = plt.subplots(constrained_layout=True)
    ax.semilogy(vals, empirical, "o", color="black", label="empirical")
    ax.semilogy(
        vals,
        scaled.probabilities,
        "-",
        color="#d62728",
        label=fr"scaled Omega MLE, $a={scaled_a:.3g}$",
    )
    ax.semilogy(vals, pure.probabilities, "--", color="#1f77b4", alpha=0.72, label="pure Omega")
    ax.set_xlabel(r"Elias Omega codelength $\ell_\Omega$")
    ax.set_ylabel(r"Probability mass on observed support")
    ax.set_title(f"{label}: Omega tail-regime diagnostic")
    ax.legend(frameon=False)
    fig.savefig(fitted_path, bbox_inches="tight", metadata=metadata)
    if show:
        plt.show()
    plt.close(fig)

    return prior_path, fitted_path


def analyze_sizes(
    sizes: Iterable[int],
    label: str,
    output_prefix: str | Path,
    n_bootstrap: int = 1000,
    seed: int = 20240524,
    tail_drop: int = 0,
    metrics_csv: str | Path | None = None,
    show: bool = False,
) -> AnalysisResult:
    raw_vals, raw_counts = histogram_from_sizes(sizes)
    vals, counts = select_tail_support(raw_vals, raw_counts, tail_drop=tail_drop)
    models = fit_omega_models(vals, counts)
    ci = bootstrap_scaled_omega(vals, counts, n_bootstrap=n_bootstrap, seed=seed)
    prior_path, fitted_path = save_figures(label, vals, counts, models, output_prefix, show=show)
    metrics_path = Path(metrics_csv) if metrics_csv is not None else Path(f"{output_prefix}_model_metrics.csv")
    write_metrics_csv(
        metrics_path,
        models,
        n_obs=int(np.sum(counts)),
        n_support=int(vals.size),
        min_omega=int(vals[0]),
        max_omega=int(vals[-1]),
        tail_drop=tail_drop,
        dropped_n=int(np.sum(raw_counts) - np.sum(counts)),
        bootstrap_ci=ci,
    )

    result = AnalysisResult(
        label=label,
        vals=vals,
        counts=counts,
        raw_vals=raw_vals,
        raw_counts=raw_counts,
        tail_drop=tail_drop,
        models=models,
        bootstrap_ci=ci,
        prior_figure=prior_path,
        fitted_figure=fitted_path,
        metrics_csv=metrics_path,
    )
    print_report(result)
    return result
