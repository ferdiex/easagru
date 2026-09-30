"""
easagru_stats.py - rebuild the numbers of the paper's tables and statistical
tests from the result files in runs/ (no simulation).

  Table 3  small arena: test fitness with the signal normal / off / random,
           and signal specificity (from the diagnosis files)
  Table 4  BG priorities of the evolved EASA champions (decoded from champion.npy)
  Table 5  big arena: the three replicates evolved there
  Tests    paired Wilcoxon tests on the same test episodes, and Welch t-test
           between conditions over replicate means

Output: printed, and saved in results_tables/ (Markdown and CSV).

Usage:  python3 easagru_stats.py
"""
import csv
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "results_tables")

SMALL = ["gru_s1", "gru_s2", "gru_s3", "easa_evo_s1", "easa_evo_s2", "easa_evo_s3",
         "easa_safe_s1", "easa_s1", "gru_compass_s1"]
BIG = ["gru_big_s1", "gru_big_s2", "gru_big_s3"]
MODES = ["normal", "off", "random"]


def eval_file(run):
    for name in (f"evaluation_{run}.csv", "evaluation.csv"):
        p = os.path.join(RUNS, run, name)
        if os.path.exists(p):
            return p
    return None


def load_eval(run):
    p = eval_file(run)
    if p is None:
        return None
    by = {}
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            by.setdefault(r["signal"], {})[int(r["seed"])] = float(r["fitness"])
    return by


def specificity(run):
    for name in (f"diagnosis_{run}.txt", "diagnosis_summary.txt"):
        p = os.path.join(RUNS, run, name)
        if os.path.exists(p):
            t = open(p).read()
            on = re.search(r"P\(emit \| on food\)\s+([\d.]+)%", t)
            off = re.search(r"P\(emit \| off food\)\s+([\d.]+)%", t)
            return (float(on.group(1)) if on else np.nan, float(off.group(1)) if off else np.nan)
    return (np.nan, np.nan)


def paired(a, b):
    from scipy.stats import wilcoxon
    seeds = sorted(set(a) & set(b))
    x = np.array([a[s] for s in seeds]); y = np.array([b[s] for s in seeds])
    d = x - y
    p = wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
    return d.mean(), p, len(seeds)


def table_rows(runs):
    rows = []
    for run in runs:
        ev = load_eval(run)
        if ev is None:
            print(f"  (skip {run}: no evaluation file)")
            continue
        m = [np.mean(list(ev[k].values())) if k in ev else np.nan for k in MODES]
        on, off = specificity(run)
        rows.append([run] + [f"{v:.3f}" for v in m] + [f"{on:.1f}%", f"{off:.1f}%"])
    return rows


def write(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    text = "\n".join(md)
    with open(os.path.join(OUT, name + ".md"), "w") as f:
        f.write(text + "\n")
    print(f"\n{name}\n{text}")


def main():
    header = ["run", "normal", "off", "random", "P(emit|on food)", "P(emit|off food)"]
    write("table3_small_arena", header, table_rows(SMALL))
    write("table5_big_arena", header, table_rows(BIG))

    # Table 4: evolved BG priorities
    try:
        from easagru_brain import ResGRU, VARIANTS, decode_bg_genes
        rows = []
        v = VARIANTS["safe"]
        labels = [("avoid", "obstacle"), ("gru", "obstacle"), ("explore", "hunger"), ("avoid", "gru_bid")]
        champs = {r: os.path.join(RUNS, r, "champion.npy") for r in ["easa_evo_s1", "easa_evo_s2", "easa_evo_s3"]}
        dec = {r: decode_bg_genes(np.load(p)[ResGRU.N_PARAMS:]) for r, p in champs.items() if os.path.exists(p)}
        for ch, se in labels:
            i, j = v["channels"].index(ch), v["sensors"].index(se)
            rows.append([f"{ch} <- {se}", f"{v['sw'][i][j]:+.2f}"] + [f"{dec[r][0][i, j]:+.2f}" for r in dec])
        rows.append(["dopamine D1, D2", "0.20, 0.20"] + [f"{dec[r][2]:.2f}, {dec[r][3]:.2f}" for r in dec])
        write("table4_bg_priorities", ["parameter", "hand-set"] + list(dec), rows)
    except Exception as e:  # noqa: BLE001
        print("  (Table 4 skipped:", e, ")")

    # Statistical tests
    lines = []
    ev = {r: load_eval(r) for r in SMALL + BIG}
    for r in SMALL + BIG:
        e = ev.get(r)
        if e and all(k in e for k in MODES):
            for a, b in (("normal", "off"), ("normal", "random")):
                d, p, n = paired(e[a], e[b])
                lines.append(f"{r:<15} {a}-{b:<7} diff {d:+.3f}  Wilcoxon p = {p:.2g}  (n = {n})")
    for group in (["gru_s1", "gru_s2", "gru_s3"], BIG):
        for i in range(len(group)):
            for k in range(i + 1, len(group)):
                a, b = ev.get(group[i]), ev.get(group[k])
                if a and b:
                    d, p, n = paired(a["normal"], b["normal"])
                    lines.append(f"{group[i]} vs {group[k]}  normal diff {d:+.3f}  Wilcoxon p = {p:.2g}")
    g = [np.mean(list(ev[r]["normal"].values())) for r in ["gru_s1", "gru_s2", "gru_s3"] if ev.get(r)]
    e = [np.mean(list(ev[r]["normal"].values())) for r in ["easa_evo_s1", "easa_evo_s2", "easa_evo_s3"] if ev.get(r)]
    if len(g) == 3 and len(e) == 3:
        from scipy.stats import ttest_ind, wilcoxon
        t = ttest_ind(g, e, equal_var=False)
        lines.append(f"GRU vs EASA evolved, replicate means {np.mean(g):.3f}+-{np.std(g, ddof=1):.3f} vs "
                     f"{np.mean(e):.3f}+-{np.std(e, ddof=1):.3f}, Welch t p = {t.pvalue:.2g}")
        seeds = sorted(set.intersection(*[set(ev[r]['normal']) for r in
                                          ['gru_s1', 'gru_s2', 'gru_s3', 'easa_evo_s1', 'easa_evo_s2', 'easa_evo_s3']]))
        gm = np.array([np.mean([ev[r]["normal"][s] for r in ["gru_s1", "gru_s2", "gru_s3"]]) for s in seeds])
        em = np.array([np.mean([ev[r]["normal"][s] for r in ["easa_evo_s1", "easa_evo_s2", "easa_evo_s3"]]) for s in seeds])
        lines.append(f"GRU vs EASA evolved, paired over {len(seeds)} episodes (mean of 3 runs): diff "
                     f"{(gm - em).mean():+.3f}, Wilcoxon p = {wilcoxon(gm, em).pvalue:.2g}")
    text = "\n".join(lines)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "statistical_tests.txt"), "w") as f:
        f.write(text + "\n")
    print("\nstatistical tests\n" + text)
    print(f"\n[saved] {OUT}/")


if __name__ == "__main__":
    main()
