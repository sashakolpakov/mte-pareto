#!/usr/bin/env python3
"""Analyze CRAN source package archive sizes with finite-support Omega diagnostics."""

from __future__ import annotations

import argparse
import gzip
import random
from urllib.request import Request, urlopen


DEFAULT_MIRROR = "https://cloud.r-project.org"


def fetch_bytes(url: str, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": "mte-pareto-analysis/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_content_length(url: str, timeout: float) -> int | None:
    request = Request(url, method="HEAD", headers={"User-Agent": "mte-pareto-analysis/1.0"})
    with urlopen(request, timeout=timeout) as response:
        value = response.headers.get("Content-Length")
    if value is None:
        return None
    try:
        size = int(value)
    except ValueError:
        return None
    return size if size > 0 else None


def fetch_cran_packages_index(mirror: str, timeout: float) -> str:
    url = f"{mirror.rstrip('/')}/src/contrib/PACKAGES.gz"
    payload = fetch_bytes(url, timeout=timeout)
    return gzip.decompress(payload).decode("utf-8", errors="replace")


def parse_cran_packages_index(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    package: str | None = None
    version: str | None = None

    for line in text.splitlines() + [""]:
        if not line.strip():
            if package and version:
                rows.append((package, version))
            package = None
            version = None
            continue
        if line.startswith("Package:"):
            package = line.split(":", 1)[1].strip()
        elif line.startswith("Version:"):
            version = line.split(":", 1)[1].strip()

    return rows


def choose_packages(
    packages: list[tuple[str, str]],
    max_packages: int,
    seed: int,
) -> list[tuple[str, str]]:
    packages = sorted(packages)
    if max_packages <= 0 or max_packages >= len(packages):
        return packages
    rng = random.Random(seed)
    sample = rng.sample(packages, max_packages)
    return sorted(sample)


def fetch_cran_source_sizes(
    mirror: str = DEFAULT_MIRROR,
    max_packages: int = 750,
    seed: int = 20240524,
    timeout: float = 30.0,
) -> tuple[list[int], int]:
    index = fetch_cran_packages_index(mirror=mirror, timeout=timeout)
    packages = choose_packages(
        parse_cran_packages_index(index),
        max_packages=max_packages,
        seed=seed,
    )

    sizes: list[int] = []
    for idx, (package, version) in enumerate(packages, start=1):
        url = f"{mirror.rstrip('/')}/src/contrib/{package}_{version}.tar.gz"
        try:
            size = fetch_content_length(url, timeout=timeout)
        except Exception as exc:  # pragma: no cover - depends on live CRAN mirrors
            print(f"  [!] Skipping {package!r}: {exc}")
            continue
        if size is not None:
            sizes.append(size)
        else:
            print(f"  [!] Skipping {package!r}: no Content-Length")

        if idx % 100 == 0:
            print(f"  processed {idx}/{len(packages)} packages; sizes so far: {len(sizes)}")

    return sizes, len(packages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run finite-support Omega diagnostics on CRAN source package archive sizes."
    )
    parser.add_argument("--mirror", default=DEFAULT_MIRROR, help="CRAN mirror URL")
    parser.add_argument(
        "--max-packages",
        type=int,
        default=750,
        help="deterministic sample size; use 0 for all current CRAN packages",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap replicates")
    parser.add_argument("--seed", type=int, default=20240524, help="sampling/bootstrap RNG seed")
    parser.add_argument(
        "--tail-drop",
        type=int,
        default=6,
        help="number of smallest observed Omega-codelength bins to drop before fitting",
    )
    parser.add_argument("--output-prefix", default="cran", help="prefix for output PNG/CSV files")
    parser.add_argument("--metrics-csv", default=None, help="optional metrics CSV path")
    parser.add_argument("--sizes-csv", default=None, help="optional raw size CSV path")
    parser.add_argument(
        "--use-sizes-csv",
        action="store_true",
        help="read --sizes-csv instead of querying CRAN",
    )
    parser.add_argument("--show", action="store_true", help="also display figures interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from omega_models import analyze_sizes, read_size_csv, write_size_csv

    if args.use_sizes_csv:
        if args.sizes_csv is None:
            raise SystemExit("--use-sizes-csv requires --sizes-csv")
        print(f"Reading CRAN source archive sizes from {args.sizes_csv}...")
        sizes = read_size_csv(args.sizes_csv)
        sampled_n = len(sizes)
        label = f"CRAN source archives (cached {len(sizes)} files)"
    else:
        sample_text = "all" if args.max_packages <= 0 else str(args.max_packages)
        print(f"Fetching CRAN package index and sampling {sample_text} source packages...")
        sizes, sampled_n = fetch_cran_source_sizes(
            mirror=args.mirror,
            max_packages=args.max_packages,
            seed=args.seed,
            timeout=args.timeout,
        )
        if args.sizes_csv is not None:
            path = write_size_csv(args.sizes_csv, sizes)
            print(f"Saved raw sizes: {path}")
        label = f"CRAN source archives ({len(sizes)}/{sampled_n} sampled packages)"

    print(f"Collected {len(sizes)} CRAN source archive sizes. Running Omega-tail diagnostic...\n")
    analyze_sizes(
        sizes,
        label=label,
        output_prefix=args.output_prefix,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        tail_drop=args.tail_drop,
        metrics_csv=args.metrics_csv,
        show=args.show,
    )


if __name__ == "__main__":
    main()
