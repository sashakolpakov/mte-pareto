#!/usr/bin/env python3
"""Analyze Debian package/source file sizes with finite-support Omega diagnostics."""

from __future__ import annotations

import argparse
import gzip
import lzma
from urllib.request import Request, urlopen

SOURCE_INDEX_URL = "https://deb.debian.org/debian/dists/{suite}/{component}/source/Sources.{ext}"
BINARY_INDEX_URL = "https://deb.debian.org/debian/dists/{suite}/{component}/binary-{arch}/Packages.{ext}"


def fetch_url(url: str, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": "mte-pareto-analysis/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _decompress_source_index(payload: bytes, ext: str) -> str:
    if ext == "gz":
        raw = gzip.decompress(payload)
    elif ext == "xz":
        raw = lzma.decompress(payload)
    else:
        raise ValueError(f"Unsupported Debian Sources compression: {ext}")
    return raw.decode("utf-8", errors="replace")


def fetch_debian_sources_index(
    suite: str = "stable",
    component: str = "main",
    timeout: float = 60.0,
    compression: str = "auto",
) -> tuple[str, str]:
    """Fetch and decompress a Debian Sources index."""
    extensions = ("xz", "gz") if compression == "auto" else (compression,)
    errors: list[str] = []

    for ext in extensions:
        url = SOURCE_INDEX_URL.format(suite=suite, component=component, ext=ext)
        try:
            return _decompress_source_index(fetch_url(url, timeout), ext), url
        except Exception as exc:  # pragma: no cover - depends on live Debian mirrors
            errors.append(f"{url}: {exc}")

    raise RuntimeError("Could not fetch Debian Sources index:\n" + "\n".join(errors))


def fetch_debian_binary_index(
    suite: str = "stable",
    component: str = "main",
    arch: str = "amd64",
    timeout: float = 60.0,
    compression: str = "auto",
) -> tuple[str, str]:
    """Fetch and decompress a Debian binary Packages index."""
    extensions = ("xz", "gz") if compression == "auto" else (compression,)
    errors: list[str] = []

    for ext in extensions:
        url = BINARY_INDEX_URL.format(suite=suite, component=component, arch=arch, ext=ext)
        try:
            return _decompress_source_index(fetch_url(url, timeout), ext), url
        except Exception as exc:  # pragma: no cover - depends on live Debian mirrors
            errors.append(f"{url}: {exc}")

    raise RuntimeError("Could not fetch Debian Packages index:\n" + "\n".join(errors))


def _iter_debian_control_paragraphs(text: str) -> list[dict[str, list[str]]]:
    paragraphs: list[dict[str, list[str]]] = []
    paragraph: dict[str, list[str]] = {}
    current_key: str | None = None

    for line in text.splitlines():
        if not line.strip():
            if paragraph:
                paragraphs.append(paragraph)
            paragraph = {}
            current_key = None
            continue

        if line[0].isspace() and current_key is not None:
            paragraph[current_key].append(line.strip())
            continue

        if ":" not in line:
            current_key = None
            continue

        key, value = line.split(":", 1)
        paragraph[key] = [value.strip()] if value.strip() else []
        current_key = key

    if paragraph:
        paragraphs.append(paragraph)
    return paragraphs


def _matches_file_kind(filename: str, file_kind: str) -> bool:
    if file_kind == "all":
        return True
    if file_kind == "dsc":
        return filename.endswith(".dsc")
    if file_kind == "archives":
        return not filename.endswith(".dsc")
    raise ValueError(f"Unsupported file kind: {file_kind}")


def parse_debian_source_file_sizes(text: str, file_kind: str = "all") -> list[int]:
    """Extract byte sizes from the Files field of Debian source stanzas."""
    sizes: list[int] = []
    for paragraph in _iter_debian_control_paragraphs(text):
        for file_line in paragraph.get("Files", []):
            parts = file_line.split()
            if len(parts) < 3:
                continue
            _, size_text, filename = parts[0], parts[1], parts[2]
            if not _matches_file_kind(filename, file_kind):
                continue
            try:
                size = int(size_text)
            except ValueError:
                continue
            if size > 0:
                sizes.append(size)
    return sizes


def parse_debian_binary_package_sizes(text: str) -> list[int]:
    """Extract installed archive byte sizes from Debian binary package stanzas."""
    sizes: list[int] = []
    for paragraph in _iter_debian_control_paragraphs(text):
        size_values = paragraph.get("Size", [])
        if not size_values:
            continue
        try:
            size = int(size_values[0])
        except ValueError:
            continue
        if size > 0:
            sizes.append(size)
    return sizes


def fetch_debian_source_file_sizes(
    suite: str = "stable",
    component: str = "main",
    timeout: float = 60.0,
    compression: str = "auto",
    file_kind: str = "all",
) -> tuple[list[int], str]:
    text, url = fetch_debian_sources_index(
        suite=suite,
        component=component,
        timeout=timeout,
        compression=compression,
    )
    return parse_debian_source_file_sizes(text, file_kind=file_kind), url


def fetch_debian_binary_package_sizes(
    suite: str = "stable",
    component: str = "main",
    arch: str = "amd64",
    timeout: float = 60.0,
    compression: str = "auto",
) -> tuple[list[int], str]:
    text, url = fetch_debian_binary_index(
        suite=suite,
        component=component,
        arch=arch,
        timeout=timeout,
        compression=compression,
    )
    return parse_debian_binary_package_sizes(text), url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run finite-support Omega diagnostics on Debian package/source file sizes."
    )
    parser.add_argument(
        "--dataset",
        choices=["binary", "source"],
        default="binary",
        help="Debian dataset: binary package archive sizes or source-index file sizes",
    )
    parser.add_argument("--suite", default="stable", help="Debian suite, e.g. stable or testing")
    parser.add_argument("--component", default="main", help="Debian component, e.g. main")
    parser.add_argument("--arch", default="amd64", help="Debian binary package architecture")
    parser.add_argument(
        "--compression",
        choices=["auto", "xz", "gz"],
        default="auto",
        help="Sources index compression to fetch",
    )
    parser.add_argument(
        "--file-kind",
        choices=["all", "archives", "dsc"],
        default="all",
        help="which source-index files to include",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap replicates")
    parser.add_argument("--seed", type=int, default=20240524, help="bootstrap RNG seed")
    parser.add_argument(
        "--tail-drop",
        type=int,
        default=5,
        help="number of smallest observed Omega-codelength bins to drop before fitting",
    )
    parser.add_argument("--output-prefix", default=None, help="prefix for output PNG/CSV files")
    parser.add_argument("--metrics-csv", default=None, help="optional metrics CSV path")
    parser.add_argument(
        "--sizes-csv",
        default=None,
        help="raw size CSV path",
    )
    parser.add_argument(
        "--use-sizes-csv",
        action="store_true",
        help="read --sizes-csv instead of querying Debian",
    )
    parser.add_argument("--show", action="store_true", help="also display figures interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from omega_models import analyze_sizes, read_size_csv, write_size_csv

    output_prefix = args.output_prefix or ("debian" if args.dataset == "binary" else "debian_source")
    sizes_csv = args.sizes_csv or (
        "debian_binary_package_sizes.csv"
        if args.dataset == "binary"
        else "debian_source_file_sizes.csv"
    )

    if args.use_sizes_csv:
        print(f"Reading Debian {args.dataset} sizes from {sizes_csv}...")
        sizes = read_size_csv(sizes_csv)
        source_label = f"cached {len(sizes)} files"
    else:
        if args.dataset == "binary":
            print(
                f"Fetching Debian Packages index for "
                f"{args.suite}/{args.component}/binary-{args.arch} ({args.compression})..."
            )
            sizes, url = fetch_debian_binary_package_sizes(
                suite=args.suite,
                component=args.component,
                arch=args.arch,
                timeout=args.timeout,
                compression=args.compression,
            )
        else:
            print(
                f"Fetching Debian Sources index for "
                f"{args.suite}/{args.component} ({args.compression})..."
            )
            sizes, url = fetch_debian_source_file_sizes(
                suite=args.suite,
                component=args.component,
                timeout=args.timeout,
                compression=args.compression,
                file_kind=args.file_kind,
            )
        source_label = url
        path = write_size_csv(sizes_csv, sizes)
        print(f"Saved raw sizes: {path}")

    print(f"Collected {len(sizes)} Debian {args.dataset} sizes. Running Omega-tail diagnostic...\n")
    file_kind = f", {args.file_kind}" if args.dataset == "source" else f", {args.arch}"
    analyze_sizes(
        sizes,
        label=f"Debian {args.dataset} sizes ({source_label}{file_kind})",
        output_prefix=output_prefix,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        tail_drop=args.tail_drop,
        metrics_csv=args.metrics_csv,
        show=args.show,
    )


if __name__ == "__main__":
    main()
