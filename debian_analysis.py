#!/usr/bin/env python3
"""
debian_sizes_analysis.py

Fetches Debian .deb package sizes from the stable repository and analyzes
their Elias Ω code‐length distribution against uniform, pure Ω, and fitted
scaled Ω priors. Prints KL divergences and fitted parameters, and produces
two log–log plots.
"""

import requests
import gzip
from io import BytesIO
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

# ─── Fetch Debian package sizes ───────────────────────────────────────────────
def fetch_debian_package_sizes(dist='stable', component='main', arch='amd64'):
    """
    Downloads the Packages.gz index for the given Debian distribution/component/arch,
    extracts all "Size: X" fields, and returns a list of sizes (in bytes).
    """
    url = f'https://ftp.debian.org/debian/dists/{dist}/{component}/binary-{arch}/Packages.gz'
    resp = requests.get(url)
    resp.raise_for_status()
    # Decompress the .gz content
    raw = gzip.decompress(resp.content)
    text = raw.decode('utf-8', errors='ignore')
    sizes = []
    for line in text.splitlines():
        if line.startswith('Size:'):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                sizes.append(int(parts[1]))
    return sizes

# ─── KL divergence ────────────────────────────────────────────────────────────
def kl_div(p, q):
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

# ─── Analysis & plotting ─────────────────────────────────────────────────────
def analyze_sizes(sizes, cutoff_idx=5):
    # 1) compute ℓΩ for each package size
    ints   = [int(s) for s in sizes]
    omegas = np.array([l_omega(x) for x in ints], dtype=int)

    # 2) empirical histogram of ℓΩ-values
    counts = Counter(omegas)
    vals   = np.array(sorted(counts.keys()), dtype=int)
    freqs  = np.array([counts[v] for v in vals], dtype=float)
    obs_p  = freqs / freqs.sum()

    # 3) uniform prior over distinct ℓΩ-values
    uni_p = np.ones_like(vals, dtype=float) / len(vals)

    # 4) pure Ω-prior ∝ 2^{-ℓΩ}
    phys_raw = 2.0 ** (-vals)
    phys_p   = phys_raw / phys_raw.sum()

    # 5) cut off small ℓΩ for clean log–log plotting
    vals_cut = vals[cutoff_idx:]
    obs_cut  = obs_p[cutoff_idx:]
    uni_cut  = uni_p[cutoff_idx:]
    phys_cut = phys_p[cutoff_idx:]

    # 6) KL divergences
    print(f"KL(obs ‖ pure Ω)    = {kl_div(obs_cut, phys_cut):.4f}")
    print(f"KL(obs ‖ uniform)   = {kl_div(obs_cut, uni_cut):.4f}")

    # 7) fit y = a x + c to x=ℓΩ, y=–log(obs)
    x = vals_cut
    y = -np.log(obs_cut)
    a, c = np.polyfit(x, y, 1)
    print(f"Fitted scaled Ω:     slope a = {a:.3f}, intercept c = {c:.3f}")

    # 8) build the scaled Ω-prior
    scaled_unn = np.exp(-(a * vals_cut + c))
    scaled_p   = scaled_unn / scaled_unn.sum()
    print(f"KL(obs ‖ scaled Ω)  = {kl_div(obs_cut, scaled_p):.4f}")

    # 9) Plot: observed vs. pure priors
    plt.figure(figsize=(8,5))
    plt.loglog(vals_cut, obs_cut,    '.', label='Observed')
    plt.loglog(vals_cut, phys_cut,   '.', label='Pure Ω-prior')
    plt.loglog(vals_cut, uni_cut,    '.', label='Uniform')
    plt.xlabel('ℓΩ(package size)')
    plt.ylabel('Probability')
    plt.title('Debian .deb Package Sizes vs. Priors')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # 10) Plot: including scaled Ω-prior
    plt.figure(figsize=(8,5))
    plt.loglog(vals_cut, obs_cut,     '.', label='Observed')
    plt.loglog(vals_cut, phys_cut,    '.', label='Pure Ω-prior')
    plt.loglog(vals_cut, scaled_p,    '.', label='Scaled Ω-prior')
    plt.loglog(vals_cut, uni_cut,     '.', label='Uniform')
    plt.xlabel('ℓΩ(package size)')
    plt.ylabel('Probability')
    plt.title('Debian .deb Package Sizes with Scaled Ω Fit')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Fetching Debian package sizes…")
    sizes = fetch_debian_package_sizes(dist='stable', component='main', arch='amd64')
    print(f"Collected {len(sizes)} package sizes.")
    print("Running Elias Ω analysis against priors:\n")
    analyze_sizes(sizes)
