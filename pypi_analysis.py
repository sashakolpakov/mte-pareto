#!/usr/bin/env python3
"""Analyze PyPI latest-release file sizes with finite-support Omega diagnostics."""

from __future__ import annotations

import argparse
import json
from urllib.request import Request, urlopen


TOP_PYPI_URL = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.json"


def fetch_json(url: str, timeout: float) -> dict:
    request = Request(url, headers={"User-Agent": "mte-pareto-analysis/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_top_pypi_projects(max_packages: int = 750, timeout: float = 30.0) -> list[str]:
    rows = fetch_json(TOP_PYPI_URL, timeout=timeout).get("rows", [])[:max_packages]
    return [row["project"] for row in rows if "project" in row]


def fetch_pypi_release_sizes(projects: list[str], timeout: float = 30.0) -> list[int]:
    sizes: list[int] = []
    for idx, name in enumerate(projects, start=1):
        try:
            info = fetch_json(f"https://pypi.org/pypi/{name}/json", timeout=timeout)
            for fileinfo in info.get("urls", []):
                size = fileinfo.get("size")
                if isinstance(size, int) and size > 0:
                    sizes.append(size)
        except Exception as exc:  # pragma: no cover - depends on live PyPI state
            print(f"  [!] Skipping {name!r}: {exc}")

        if idx % 100 == 0:
            print(f"  processed {idx}/{len(projects)} projects; sizes so far: {len(sizes)}")
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run finite-support Omega diagnostics on PyPI release file sizes."
    )
    parser.add_argument("--max-packages", type=int, default=750, help="top PyPI projects to query")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds")
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap replicates")
    parser.add_argument("--seed", type=int, default=20240524, help="bootstrap RNG seed")
    parser.add_argument(
        "--tail-drop",
        type=int,
        default=7,
        help="number of smallest observed Omega-codelength bins to drop before fitting",
    )
    parser.add_argument("--output-prefix", default="pypi", help="prefix for output PNG/CSV files")
    parser.add_argument("--metrics-csv", default=None, help="optional metrics CSV path")
    parser.add_argument("--sizes-csv", default=None, help="optional raw size CSV path")
    parser.add_argument(
        "--use-sizes-csv",
        action="store_true",
        help="read --sizes-csv instead of querying PyPI",
    )
    parser.add_argument("--show", action="store_true", help="also display figures interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from omega_models import analyze_sizes, read_size_csv, write_size_csv

    projects: list[str] = []
    if args.use_sizes_csv:
        if args.sizes_csv is None:
            raise SystemExit("--use-sizes-csv requires --sizes-csv")
        print(f"Reading PyPI file sizes from {args.sizes_csv}...")
        sizes = read_size_csv(args.sizes_csv)
    else:
        print(f"Fetching top {args.max_packages} PyPI projects...")
        projects = fetch_top_pypi_projects(max_packages=args.max_packages, timeout=args.timeout)
        print(f"Found {len(projects)} projects. Pulling latest-release file sizes...")
        sizes = fetch_pypi_release_sizes(projects, timeout=args.timeout)
        if args.sizes_csv is not None:
            path = write_size_csv(args.sizes_csv, sizes)
            print(f"Saved raw sizes: {path}")

    print(f"Collected {len(sizes)} file sizes. Running Omega-tail diagnostic...\n")
    label = (
        f"PyPI latest-release files (cached {len(sizes)} files)"
        if args.use_sizes_csv
        else f"PyPI latest-release files (top {len(projects)} projects)"
    )
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
