# Multiplicative Turing Ensembles, Pareto's Law, and Creativity

Empirical support code for the manuscript. The scripts fetch software-size
datasets, convert byte sizes to Elias Omega codelengths, and run a compact
finite-support diagnostic for the resulting codelength histograms.

## Installation

```bash
git clone https://github.com/sashakolpakov/mte-pareto.git
cd mte-pareto
python -m pip install numpy matplotlib
```

## Revised Analysis

The scaled Omega model is now fit by multinomial likelihood on the observed
codelength support:

```text
q_a(ell) = exp(-a ell) / sum_{ell in support} exp(-a ell)
```

The normalizing constant is determined by the support, so there is no separate
free intercept. Each run reports KL divergence and log-likelihood for the pure
Omega slope `a = log 2` and the fitted scaled-Omega slope. It also reports
bootstrap confidence intervals for the fitted slope and KL.

The empirical target is deliberately narrow: `a < log 2` is evidence for a
heavier-than-pure-Omega tail inside the Omega energy scale. These scripts are
targeted validation of that diagnostic, not a search for the best distributional
model of Debian or PyPI file sizes.

## Usage

### PyPI Latest-Release File Sizes

Fetch the top PyPI projects and analyze latest-release distribution files:

```bash
python pypi_analysis.py --max-packages 750 --bootstrap 1000 --sizes-csv pypi_file_sizes.csv
```

Use cached sizes for a deterministic rerun:

```bash
python pypi_analysis.py --use-sizes-csv --sizes-csv pypi_file_sizes.csv --bootstrap 1000
```

### Debian Binary Package Sizes

This reproduces the manuscript's Debian `.deb` package-size analysis:

```bash
python debian_analysis.py --dataset binary --suite stable --component main --arch amd64 --bootstrap 1000
```

Use cached sizes:

```bash
python debian_analysis.py --dataset binary --use-sizes-csv --sizes-csv debian_binary_package_sizes.csv --bootstrap 1000
```

### Debian Source File Sizes

This is an additional socio-technical software-size dataset from Debian
`Sources` indices:

```bash
python debian_analysis.py --dataset source --file-kind all --output-prefix debian_source --bootstrap 1000
```

`--file-kind archives` excludes `.dsc` control files; `--file-kind dsc` keeps
only source-control files.

## Outputs

Each run prints an Omega-tail diagnostic table and writes:

- `<prefix>_model_metrics.csv`
- `<prefix>_sizes_priors.png`
- `<prefix>_sizes_prior_scaled.png`
- an optional raw size cache if `--sizes-csv` is supplied

Figures are saved deterministically. Add `--show` to display them interactively.
