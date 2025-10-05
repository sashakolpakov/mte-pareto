# Multiplicative Turing Ensembles, Pareto's Law, and Creativity

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/mte-pareto.git
cd mte-pareto
```

Install dependencies:
```bash
pip install requests numpy matplotlib
```

## Usage

### PyPI Analysis
Analyzes package sizes from the top PyPI packages:
```bash
python pypi_analysis.py
```

### Debian Analysis
Analyzes package sizes from Debian stable repository:
```bash
python debian_analysis.py
```

## Results

Below we can see an empirical comparison between the actual PyPI package lengths distribution, the Gibbs prior (with self-delimiting Elias' length), and the uniform baseline (left). The rescaled Gibbs prior matches the empirical distribution quite closely (right). These two figures are output by `pypi_analysis.py`

<div align="center">
<table>
<tr>
<td><img src="pypi_sizes_priors.png" alt="PyPI Sizes vs Gibbs / Uniform" width="400"/></td>
<td><img src="pypi_sizes_prior_scaled.png" alt="PyPI Sizes with Scaled Gibbs Prior" width="400"/></td>
</tr>
</table>
</div>
