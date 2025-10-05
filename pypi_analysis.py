import requests
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# ─── Elias Ω code‐length ───────────────────────────────────────────────────────
def l_omega(n: int) -> int:
    """
    Elias Ω code length for integer n.
    ℓΩ(1) = 1.
    For n > 1: ℓΩ(n) = 1 + sum_{k = n, floor(log2 k), … > 1} floor(log2 k).
    """
    length = 1
    k = n
    while k > 1:
        bits = k.bit_length()       # = floor(log2 k) + 1
        length += bits - 1          # add floor(log2 k)
        k = bits - 1                # next k
    return length

# ─── Fetch top PyPI packages (last 30 days) ────────────────────────────────────
def fetch_top_pypi_projects(max_packages: int = 100):
    url = 'https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.json'
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    rows = data.get('rows', [])[:max_packages]
    return [r['project'] for r in rows]

# ─── Fetch each project's latest release file sizes ───────────────────────────
def fetch_pypi_repo_sizes(projects):
    sizes = []
    for name in projects:
        try:
            resp = requests.get(f'https://pypi.org/pypi/{name}/json')
            resp.raise_for_status()
            info = resp.json()
            # 'urls' holds every built distribution for the latest release
            for fileinfo in info.get('urls', []):
                size = fileinfo.get('size')
                if isinstance(size, int) and size > 0:
                    sizes.append(size)
        except Exception as e:
            print(f"  [!] Skipping {name!r} due to error: {e}")
    return sizes

# ─── KL divergence ────────────────────────────────────────────────────────────
def kl_div(p, q):
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

# ─── Analysis & plotting ─────────────────────────────────────────────────────
def analyze_sizes(sizes, cutoff_idx=7):
    # 1) empirical ℓΩ histogram
    ints = [int(s) for s in sizes]
    omegas = [l_omega(x) for x in ints]
    counts = Counter(omegas)
    vals  = np.array(sorted(counts.keys()), dtype=int)
    freqs = np.array([counts[v] for v in vals], dtype=float)
    obs_p = freqs / freqs.sum()

    # 2) uniform prior over distinct ℓΩ‐values
    uni_p = np.ones_like(vals, dtype=float) / len(vals)

    # 3) pure Ω‐prior ∝ 2^{-ℓΩ}
    phys_raw = 2.0 ** (-vals)
    phys_p   = phys_raw / phys_raw.sum()

    # 4) cut off the very-small side for clean log–log
    vals_cut = vals[cutoff_idx:]
    obs_cut  = obs_p[cutoff_idx:]
    uni_cut  = uni_p[cutoff_idx:]
    phys_cut = phys_p[cutoff_idx:]

    # 5) KL divergences
    print(f"KL(obs ‖ pure Ω)   = {kl_div(obs_cut, phys_cut):.4f}")
    print(f"KL(obs ‖ uniform)  = {kl_div(obs_cut, uni_cut):.4f}")

    # 6) fit y = a x + c to x=ℓΩ, y=–log(obs)
    x = vals_cut
    y = -np.log(obs_cut)
    a, c = np.polyfit(x, y, 1)
    print(f"Fitted scaled Ω:   slope a = {a:.3f}, intercept c = {c:.3f}")

    # 7) build the scaled Ω‐prior
    scaled_unn = np.exp(-(a * vals_cut + c))
    scaled_p   = scaled_unn / scaled_unn.sum()
    print(f"KL(obs ‖ scaled Ω) = {kl_div(obs_cut, scaled_p):.4f}")

    # 8) plots
    plt.figure(figsize=(8,5))
    plt.loglog(vals_cut, obs_cut,    '.', label='Observed')
    plt.loglog(vals_cut, phys_cut,   '.', label='Pure Ω-prior')
    plt.loglog(vals_cut, uni_cut,    '.', label='Uniform')
    plt.xlabel('ℓΩ(size)')
    plt.ylabel('Probability')
    plt.title('PyPI Package Sizes vs. Priors')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.show()

    plt.figure(figsize=(8,5))
    plt.loglog(vals_cut, obs_cut,     '.', label='Observed')
    plt.loglog(vals_cut, phys_cut,    '.', label='Pure Ω-prior')
    plt.loglog(vals_cut, scaled_p,    '.', label='Scaled Ω-prior')
    plt.loglog(vals_cut, uni_cut,     '.', label='Uniform')
    plt.xlabel('ℓΩ(size)')
    plt.ylabel('Probability')
    plt.title('Including Scaled Ω-Prior Fit')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.show()

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Fetching top PyPI projects…")
    projects = fetch_top_pypi_projects(max_packages=750)
    print(f"Found {len(projects)} projects. Pulling their release sizes…")
    sizes = fetch_pypi_repo_sizes(projects)
    print(f"Collected {len(sizes)} file‐sizes (bytes). Running analysis…\n")
    analyze_sizes(sizes)
