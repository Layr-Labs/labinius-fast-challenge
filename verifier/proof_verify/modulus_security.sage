# Security of this challenge's commitment modulus.
#
# The estimator below is the procedure in solution/labinius/params.ipynb.
# This instance is a clear opening, so the length bound is the notebook's
# clear-opening call: expected_norm at corr0_999, with no dropped-bit term.
# 2^17 F162 elements are 2^15 ring elements, and the notebook writes the
# coefficient count as that times 648.
#
#   sage modulus_security.sage <product>
#   sage modulus_security.sage --table
#
# The first form prints the notebook's module-BKZ line, then the bit cost, and
# exits 0 when the cost is at least MIN_BITS. The checker does not run it.
# `--table` prints one line per non-empty subset of the seven primes,
# `product microbits`, with microbits -1 when the instance is trivially easy.
# CI diffs that against modulus_estimates.txt, which the checker compiles in.

import os
import sys

# Conda Sage copies this file under /tmp before executing it, so the estimator
# location cannot be taken from __file__. The checker passes the absolute path.
estimator_root = os.environ.get("LATTICE_ESTIMATOR")
if not estimator_root:
    estimator_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lattice-estimator")
sys.path.insert(0, estimator_root)

from estimator import *  # noqa: F403

Logging.set_level(Logging.LEVEL0)

f = 3^5 * 2^3
phi = euler_phi(f)

L1 = 2917 #Q
L2 = 3889 # FS
L3 = 4861 # Q
L4 = 9721 #FS
L5 = 12637 # Q
L6 = 17497 # L
L7 = 19441 #L

tau = 28 # weigth
gamma = 12 * sqrt(3)

# Module-BKZ correction, following Ducas--Engelberts--de Perthuis, "Predicting Module-Lattice Reduction"
# (ePrint 2025/1904).  Condensed from their predictions.py / pred_gain.py, "upper bound" setting.
#
# Idea: module-BKZ over a field K of degree d inserts a whole rank-1 module b*O_K (d vectors) per SVP call.
# How much profile volume those d vectors eat is governed by the density of O_K as a lattice, i.e. by the
# discriminant gap ln(|Delta_K| / d^d).  It is 0 for power-of-two conductors (no gain, only the d-1 penalty),
# and negative whenever an odd prime divides the conductor (flatter slope, i.e. a smaller blocksize suffices).
# An attacker on Q(zeta_c) may reduce over ANY subfield; for c = 2^a 3^b every subfield containing zeta_3 has the
# same per-degree gap, so the cheapest one, Q(zeta_3) with d = 2, is the attacker's best choice.  For Q(zeta_3)
# the paper's skewness and index corrections (their "lower estimate") vanish, so this estimate is exact within
# their model; for other c it is their "upper bound" (the more conservative of their two curves).
from functools import lru_cache
from math import log, lgamma, pi
import numpy as np

@lru_cache(maxsize=None)
def primes_of(c):
    return tuple(p for p in range(2, c + 1) if c % p == 0 and all(p % r for r in range(2, p)))

@lru_cache(maxsize=None)
def cyclo(c):
    # degree phi(c), log|Delta_K| = d ln c - sum_{p|c} d ln p/(p-1)  [Washington, Prop. 2.7], number of roots of unity
    pfs = primes_of(c)
    d = 1
    for p in pfs:
        e = 0; m = c
        while m % p == 0: m //= p; e += 1
        d *= (p - 1) * p ** (e - 1)
    ldisc = d * log(c) - sum(d * log(p) / (p - 1) for p in pfs)
    mu = 2 * c if c % 2 else c
    return d, ldisc, mu

def lgh(n):
    # log Gaussian heuristic of a unit-volume n-dim lattice (their lghZ; Section 3.1)
    return (log(2) - np.euler_gamma - (n / 2) * log(pi) + lgamma(n / 2 + 1)) / n

def slope(c, beta_K):
    # predicted Q-slope of module-BKZ over Q(zeta_c) with K-rank beta_K blocks (Eq. (3)/(4), Heuristic Claim 2)
    #   t1 = module Gaussian heuristic = lgh(d*beta_K) + ln(mu_K/2)/(d*beta_K)       [Section 4.2]
    #   t2 = ln|Delta_K|/(2d) - ln(d)/2   (the discriminant gap; the whole gain lives here)  [Section 4.3]
    #   t3 (skewness), t4 (index) dropped: this is their "upper bound", exact for d = 2   [Sections 4.4, 4.5]
    # c = 1 gives plain BKZ (d = 1, Delta = 1, mu = 2), i.e. slope = -2 lgh(beta)/(beta-1)
    d, ldisc, mu = cyclo(c)
    t = lgh(d * beta_K) + log(mu / 2) / (d * beta_K) + ldisc / (2 * d) - log(d) / 2
    return -(2 / (d * (beta_K - 1))) * t

def beta_eq(beta, c=3):
    # Z-dimension of the SVP oracle module-BKZ over Q(zeta_c) needs to reach the slope of BKZ-beta.
    # Integer search over the K-rank, then linear interpolation (same as their find_beta_eq).
    d = cyclo(c)[0]
    s = slope(1, beta)
    b = max(beta // d, 3)
    sg = 1 if slope(c, b) < s else -1
    while sg * slope(c, b) < sg * s: b += sg
    b_ = b - sg * (s - slope(c, b)) / (slope(c, b - sg) - slope(c, b))
    return d * b_

@lru_cache(maxsize=None)
def coeff_to_canonical(c):
    # NOT from the paper (their Open Question 5).  Module-BKZ must work in the canonical embedding, while the
    # SIS bound is on coefficient vectors.  Gram matrix of the trace form on the power basis of Z[zeta_c] has
    # entries Tr(zeta^(i-j)) = Ramanujan sums; its eigenvalues lam_i tell how the two norms disagree per direction.
    # For a short vector in a random direction, E||coeff||^2 = ||canon||^2 * mean(1/lam)  -> bound scales by
    # sqrt(harmonic mean); the lattice volume scales by sqrt(geometric mean) = |Delta_K|^(1/2d).
    # Returned factor = sqrt(harm/geo) multiplies the length bound; = 1 for power-of-two c, 0.931 for c = 2^a 3^b.
    d = cyclo(c)[0]
    mob = lambda m: 0 if any(m % (p * p) == 0 for p in primes_of(m)) else (-1) ** len(primes_of(m))
    tr = lambda k: mob(c // np.gcd(k, c)) * d // cyclo(c // np.gcd(k, c))[0]
    trv = [tr(k) for k in range(c)]
    lam = np.linalg.eigvalsh(np.array([[trv[(i - j) % c] for j in range(d)] for i in range(d)], dtype=float))
    return float(np.sqrt(1 / np.mean(1 / lam) / np.exp(np.mean(np.log(lam)))))

def sis_mbkz(params, c=3, ring_conductor=None, name=""):
    # 1. run the estimator's usual SIS lattice attack (optionally with the embedding-distorted bound),
    # 2. translate its blocksize to the module-BKZ equivalent over Q(zeta_c), rounded to a multiple of deg,
    # 3. re-cost with the same reduction cost model (RC.MATZOV) at the same lattice dimension d.
    # The unstructured attacker never needs the distortion, so compare mbits against the undistorted BKZ bits.
    f = coeff_to_canonical(ring_conductor) if ring_conductor else 1.0
    cost = SIS.lattice(params.updated(length_bound=params.length_bound * f))
    beta, d = int(cost["beta"]), int(cost["d"])
    dK = cyclo(c)[0]
    be = beta_eq(beta, c)
    be_r = int(round(be / dK)) * dK
    bits, mbits = float(log(cost["rop"], 2)), float(log(RC.MATZOV(be_r, d), 2))
    sys.stderr.write(f"{name:14s} BKZ: beta={beta:4d} d={d} {bits:6.1f} bits | mBKZ/Q(z{c}): beta_eq={be:6.1f} {mbits:6.1f} bits | distortion x{f:.3f}\n")
    return mbits

# observed
corr0_95 = 2.0
corr0_98 = 3.07
corr0_999 = 3.85

def expected_norm(dim,coo = corr0_999):
    return sqrt(dim * tau * coo)

def account_for_extraction(norm):
    return norm * 8 * gamma

def with_dropped_bits(norm, d, r, coo = corr0_999, height=1):
    return sqrt(norm^2 + height * phi * r * tau * (4^d - 1) / 12 * coo)

# This challenge asks for about 100 bits. The notebook reports module-BKZ bits.
MIN_BITS = 100

def module_bkz_microbits(q):
    try:
        params = SIS.Parameters(
            n=phi,
            q=q,
            length_bound=account_for_extraction(expected_norm(2^15 * 648)),
            norm=2,
        )
        mbits = sis_mbkz(params, ring_conductor=f)
    except ValueError as err:
        sys.stderr.write(f"{q}: {err}\n")
        return int(-1)
    return int(round(float(mbits) * float(10^6)))

def dump_table():
    primes = [L1, L2, L3, L4, L5, L6, L7]
    rows = []
    total = int(2^7 - 1)
    seen = int(0)
    for mask in range(int(1), int(2^7)):
        q = Integer(1)
        for i in range(int(7)):
            if mask & (int(1) << i):
                q *= Integer(primes[i])
        seen += int(1)
        micro = module_bkz_microbits(q)
        sys.stderr.write(f"[{int(seen)}/{total}] {int(q)} {int(micro)}\n")
        sys.stderr.flush()
        rows.append((q, micro))
    rows.sort(key=lambda row: row[0])
    sys.stdout.write("# product microbits\n")
    for q, micro in rows:
        sys.stdout.write(f"{int(q)} {int(micro)}\n")
    sys.stdout.flush()

def main():
    if int(phi) != int(648):
        sys.stderr.write("ring degree is not the notebook's 648\n")
        os._exit(int(2))
    if len(sys.argv) == int(2) and str(sys.argv[int(1)]) == "--table":
        dump_table()
        os._exit(int(0))
    raw = os.environ.get("LABINIUS_MODULUS")
    if not raw:
        if len(sys.argv) != int(2):
            sys.stderr.write("usage: sage modulus_security.sage <product> | --table\n")
            os._exit(int(2))
        raw = sys.argv[int(1)]
    micro = module_bkz_microbits(Integer(raw))
    sys.stdout.write(f"{float(micro) / float(10^6):.6f}\n")
    sys.stdout.flush()
    os._exit(int(0) if micro >= int(MIN_BITS) * int(10^6) else int(1))

main()
