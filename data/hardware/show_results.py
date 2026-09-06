#!/usr/bin/env python3
"""Print the hardware results table from trials_index.csv.
Plain Python — no ROS needed.  Run:  python3 show_results.py"""
import csv, collections, os
from math import comb, sqrt

D = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(D, "trials_index.csv"))))
saved = [r for r in rows if r["decision"] == "saved"]

NAMES = {"v11": "SAC-R-PV-STAM", "mlp_fs": "SAC-MLP-FS", "v8": "SAC-PV-STAM",
         "v10": "SAC-PV-STAM-H", "baseline": "SAC-MLP"}
ORDER = ["v11", "mlp_fs", "v8", "v10", "baseline"]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0, (c - h) * 100), min(100, (c + h) * 100)


def fisher(a, b, c, d):
    n = a + b + c + d
    r1, r2, c1 = a + b, c + d, a + c

    def p(x):
        y, z = r1 - x, c1 - x
        w = r2 - z
        if min(y, z, w) < 0:
            return 0.0
        return comb(r1, x) * comb(r2, z) / comb(n, c1)
    obs = p(a)
    return sum(p(x) for x in range(0, min(r1, c1) + 1) if p(x) <= obs + 1e-12)


by = collections.defaultdict(list)
for r in saved:
    by[(r["model"], r["scenario"])].append(r["outcome"])

print("=" * 74)
print("  PV-STAM HARDWARE RESULTS   —   %d valid trials" % len(saved))
print("=" * 74)
print("  %-16s %-9s %3s %7s %7s %7s   %s" %
      ("model", "scenario", "n", "goal", "coll", "timeout", "success (95% CI)"))
print("  " + "-" * 70)
for m in ORDER:
    for sc in ("2", "3"):
        o = by.get((m, sc))
        if not o:
            continue
        n = len(o)
        g, c, t = o.count("goal"), o.count("collision"), o.count("timeout")
        lo, hi = wilson(g, n)
        print("  %-16s %-9s %3d %7d %7d %7d   %5.1f%%  (%.0f-%.0f%%)" %
              (NAMES[m], "static" if sc == "2" else "moving", n, g, c, t,
               g / n * 100, lo, hi))

print()
tot = collections.Counter()
for m in ORDER:
    g = sum(o.count("goal") for (mm, _), o in by.items() if mm == m)
    n = sum(len(o) for (mm, _), o in by.items() if mm == m)
    c = sum(o.count("collision") for (mm, _), o in by.items() if mm == m)
    if n:
        lo, hi = wilson(g, n)
        print("  POOLED  %-16s %2d/%2d = %5.1f%%  (%.0f-%.0f%%)   collisions %d" %
              (NAMES[m], g, n, g / n * 100, lo, hi, c))

print()
print("  KEY COMPARISONS (Fisher exact, two-sided)")


def cell(m, sc):
    o = by[(m, sc)]
    return o.count("goal"), len(o) - o.count("goal")


tests = [
    ("v11 vs MLP-FS   pooled", sum(cell("v11", s)[0] for s in "23"), sum(cell("v11", s)[1] for s in "23"),
     sum(cell("mlp_fs", s)[0] for s in "23"), sum(cell("mlp_fs", s)[1] for s in "23")),
    ("v11 vs MLP-FS   moving", *cell("v11", "3"), *cell("mlp_fs", "3")),
    ("v11 vs MLP-FS   static", *cell("v11", "2"), *cell("mlp_fs", "2")),
    ("v11 vs PV-STAM  moving", *cell("v11", "3"), *cell("v8", "3")),
    ("v11 vs PV-STAM-H static", *cell("v11", "2"), *cell("v10", "2")),
    ("v11 vs MLP       static", *cell("v11", "2"), *cell("baseline", "2")),
    ("PV-STAM static vs moving", *cell("v8", "2"), *cell("v8", "3")),
]
for lab, a, b, c, d in tests:
    p = fisher(a, b, c, d)
    print("    %-26s p = %-12.7f %s" % (lab, p, "SIGNIFICANT" if p < 0.05 else "not significant"))

print()
ex = collections.Counter(r["decision"] for r in rows if r["decision"] != "saved")
print("  EXCLUDED: %d of %d recorded trials" % (sum(ex.values()), len(rows)))
for k, v in sorted(ex.items()):
    print("    %-22s %d   (reasons in the 'note' column)" % (k, v))
print("=" * 74)
