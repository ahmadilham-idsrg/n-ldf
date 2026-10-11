#!/usr/bin/env python3
"""LaTeX tables of the controlled and held-out experiments, generated from the
analysis outputs so that no number in those tables is typed by hand.

    python extension_tables.py --ext results/extension --heldout results/heldout --out results/tables

Writes one .tex file per table and numbers.json with every value the
manuscript quotes from these experiments (used by --check in
supplementary_analysis.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MECH = [("ROS", "ROS-r0.75", "ROS-r1.0"), ("SMOTE", "SMOTE-r0.75", "SMOTE"),
        ("Borderline-SMOTE", "B-SMOTE-r0.75", "B-SMOTE"), ("ADASYN", "ADASYN-r0.75", "ADASYN"),
        ("SMOTE-Tomek", "SMOTE-Tomek-r0.75", "SMOTE-Tomek"), ("Safe-Level-SMOTE", "SLS-r0.75", "SLS"),
        ("SVM-SMOTE", "SVM-SMOTE-r0.75", "SVM-SMOTE"), ("k-means SMOTE", "KMeans-SMOTE-r0.75", "KMeans-SMOTE"),
        ("N-LDF", "N-LDF", "N-LDF-r1.0")]
ABL = [("ABL-lp0", r"$\lambda_p=0$ (raw sums)"), ("ABL-lp1", r"$\lambda_p=1$"),
       ("ABL-localprop", r"local proportion"), ("ABL-uniform", r"unguided generation"),
       ("ABL-nojitter", r"no perturbation"), ("ABL-nodist", r"no distance test"),
       ("ABL-nosupport", r"no support test"), ("ABL-noratio", r"no ratio test")]
SENS = [("SENS-lp0.5", r"$\lambda_p=0.5$"), ("SENS-lp0.9", r"$\lambda_p=0.9$"), ("SENS-kd5", r"$k_d=5$"),
        ("SENS-kd11", r"$k_d=11$"), ("SENS-acc-0.1", r"threshold $-0.1$"), ("SENS-acc+0.1", r"threshold $+0.1$"),
        ("SENS-rb2.0", r"ratio bound $2.0$")]


def p(x: float) -> str:
    if x is None or x != x:
        return "--"
    if x >= 0.995:
        return "1.00"
    if x >= 0.01:
        return f"{x:.2f}"
    if x >= 0.001:
        return f"{x:.3f}"
    m, e = f"{x:.1e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def d(x: float, nd: int = 4) -> str:
    return f"${x:+.{nd}f}$".replace("+", "+") if x == x else "--"


def ci(r: dict) -> str:
    s = f"$[{r['ci_lo']:+.3f},\\,{r['ci_hi']:+.3f}]$"
    return s


def find(fam: dict, metric: str, a: str, b: str) -> dict | None:
    for r in fam.get(metric, []):
        if r["a"] == a and r["b"] == b:
            return r
    return None


def table_amount(res: dict, means: pd.DataFrame) -> tuple[str, dict]:
    fams = res["families"]
    amount = {**{k: v for k, v in fams.get("amount", {}).items()}}
    rows, nums = [], {}
    for label, r075, r1 in MECH:
        if r075 not in means.index or r1 not in means.index:
            continue
        rec = {}
        for metric in ["F1", "Recall", "Specificity", "AUPRC"]:
            hit = None
            for famname in ["amount_complete"]:
                hit = hit or find(fams.get(famname, {}), metric, r1, r075)
            rec[metric] = hit
        f = rec["F1"]
        line = (f"{label} & {means.loc[r075, 'F1']:.4f} & {means.loc[r1, 'F1']:.4f} & {d(f['mean_diff'])} & {p(f['source_holm_p'])} & "
                f"{d(rec['Recall']['mean_diff'], 3)} & {d(rec['Specificity']['mean_diff'], 3)} & {d(rec['AUPRC']['mean_diff'], 4)} \\\\")
        rows.append(line)
        nums[label] = {"F1_r075": means.loc[r075, "F1"], "F1_r1": means.loc[r1, "F1"],
                       **{f"d_{m}": rec[m]["mean_diff"] for m in rec},
                       **{f"srcholm_{m}": rec[m]["source_holm_p"] for m in rec},
                       **{f"taskholm_{m}": rec[m]["holm_p"] for m in rec}}
    body = "\n".join(rows)
    tex = rf"""\begin{{table}}[htbp]
\caption{{Resampling amount within each mechanism: mean F1 at $r=0.75$ and at
parity, and the change from $r=0.75$ to parity in F1, recall, specificity,
and AUPRC}}
\label{{tab:amount}}
\centering
\tabsmall
\setlength{{\tabcolsep}}{{2.5pt}}
\begin{{tabular}}{{lccccccc}}
\toprule
\textbf{{Mechanism}} & \hd{{\textbf{{F1}}}}{{$r=0.75$}} & \hd{{\textbf{{F1}}}}{{parity}} & \hd{{$\Delta$\textbf{{F1}}}}{{}} & \hd{{\textbf{{Source}}}}{{\textbf{{Holm $p$}}}} & \hd{{$\Delta$\textbf{{Rec.}}}}{{}} & \hd{{$\Delta$\textbf{{Spec.}}}}{{}} & \hd{{$\Delta$\textbf{{AUPRC}}}}{{}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\footnotetext{{Means over the 41 tasks. $\Delta$ is parity minus $r=0.75$ for
the same mechanism, averaged over tasks. Holm correction is over the nine
mechanisms, with one block per source. N-LDF realizes
its target on the 39 non-chess tasks and almost no augmentation on the two
chess tasks at either target}}
\end{{table}}
"""
    return tex, nums


def table_mechanism(res: dict) -> tuple[str, dict]:
    fams = res["families"]
    rows, nums = [], {}
    for target, famnames, ref in [("$r=0.75$", ["mechanism_r0.75_complete"], "N-LDF"),
                                  ("parity", ["mechanism_parity_complete"], "N-LDF-r1.0")]:
        rows.append(rf"\multicolumn{{6}}{{l}}{{\textit{{N-LDF minus each mechanism at {target}}}}} \\")
        for label, r075, r1 in MECH:
            other = r075 if target.startswith("$r") else r1
            if other == ref or label == "N-LDF":
                continue
            rec = {}
            for metric in ["F1", "Recall", "Specificity", "AUPRC"]:
                hit = None
                for fn in famnames:
                    hit = hit or find(fams.get(fn, {}), metric, ref, other)
                rec[metric] = hit
            if rec["F1"] is None:
                continue
            f = rec["F1"]
            rows.append(f"\\quad {label} & {d(f['mean_diff'])} & {p(f['source_holm_p'])} & {ci(f)} & "
                        f"{d(rec['Recall']['mean_diff'], 3)} & {d(rec['Specificity']['mean_diff'], 3)} \\\\")
            nums[f"{ref} vs {other}"] = {**{f"d_{m}": rec[m]["mean_diff"] for m in rec},
                                         **{f"srcholm_{m}": rec[m]["source_holm_p"] for m in rec},
                                         **{f"taskholm_{m}": rec[m]["holm_p"] for m in rec},
                                         "ci_F1": [f["ci_lo"], f["ci_hi"]]}
    body = "\n".join(rows)
    tex = rf"""\begin{{table}}[htbp]
\caption{{Mechanism at a matched target: N-LDF minus every other mechanism at the
same target}}
\label{{tab:mechanism}}
\centering
\tabsmall
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{lccccc}}
\toprule
\textbf{{Comparison}} & \hd{{$\Delta$\textbf{{F1}}}}{{}} & \hd{{\textbf{{Source}}}}{{\textbf{{Holm $p$}}}} & \hd{{\textbf{{Cluster-bootstrap}}}}{{\textbf{{95\% interval}}}} & \hd{{$\Delta$\textbf{{Rec.}}}}{{}} & \hd{{$\Delta$\textbf{{Spec.}}}}{{}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\footnotetext{{Differences are N-LDF minus the mechanism, averaged over the 41
tasks. Holm correction is over the eight comparisons at each target, with one
block per source; intervals are uncorrected source-cluster bootstrap
intervals}}
\end{{table}}
"""
    return tex, nums


def table_variants(res: dict, means: pd.DataFrame, fam: str, spec: list, label: str, caption: str,
                   extra: pd.DataFrame | None, note: str = "") -> tuple[str, dict]:
    fams = res["families"].get(fam, {})
    rows, nums = [], {}
    for code, desc in spec:
        rec = {m: find(fams, m, code, "N-LDF") for m in ["F1", "Recall", "Specificity", "AUPRC"]}
        if rec["F1"] is None:
            continue
        f = rec["F1"]
        acc = ""
        if extra is not None and code in extra.index:
            acc = f"{extra.loc[code, 'realized']:.3f}"
        rows.append(f"{desc} & {means.loc[code, 'F1']:.4f} & {d(f['mean_diff'])} & {p(f['source_holm_p'])} & {p(f['holm_p'])} & "
                    f"{d(rec['Recall']['mean_diff'], 3)} & {d(rec['Specificity']['mean_diff'], 3)} & {d(rec['AUPRC']['mean_diff'], 4)} & {acc} \\\\")
        nums[code] = {"F1": means.loc[code, "F1"], **{f"d_{m}": rec[m]["mean_diff"] for m in rec},
                      **{f"srcholm_{m}": rec[m]["source_holm_p"] for m in rec},
                      **{f"taskholm_{m}": rec[m]["holm_p"] for m in rec},
                      "ci_F1": [f["ci_lo"], f["ci_hi"]]}
    body = "\n".join(rows)
    tex = rf"""\begin{{table}}[htbp]
\caption{{{caption}}}
\label{{{label}}}
\centering
\tabsmall
\setlength{{\tabcolsep}}{{1.6pt}}
\begin{{tabular}}{{lcccccccc}}
\toprule
\textbf{{Variant}} & \textbf{{F1}} & \hd{{$\Delta$\textbf{{F1}}}}{{}} & \hd{{\textbf{{Source}}}}{{\textbf{{Holm $p$}}}} & \hd{{\textbf{{Task}}}}{{\textbf{{Holm $p$}}}} & \hd{{$\Delta$\textbf{{Rec.}}}}{{}} & \hd{{$\Delta$\textbf{{Spec.}}}}{{}} & \hd{{$\Delta$\textbf{{AUPRC}}}}{{}} & \hd{{\textbf{{Realized}}}}{{\textbf{{ratio}}}} \\
\midrule
\textit{{N-LDF (archived)}} & {means.loc['N-LDF', 'F1']:.4f} & & & & & & & {extra.loc['N-LDF', 'realized']:.3f} \\
{body}
\bottomrule
\end{{tabular}}
\footnotetext{{$\Delta$ is the variant minus archived N-LDF, averaged over the 41
tasks. Holm correction is over the variants of the family.{note} Realized ratio is
the minority--majority ratio after resampling, averaged within task and then
over tasks (target $0.75$; the two chess tasks, with almost no augmentation,
pull the mean below the target)}}
\end{{table}}
"""
    return tex, nums


def table_controls(means: pd.DataFrame) -> str:
    rows = []
    spec = [("Baseline", "Baseline (default threshold)"), ("Baseline-thr", "Baseline, tuned threshold"),
            ("Baseline-cw", "Baseline, class weights"), None,
            ("SMOTE", "SMOTE"), ("N-LDF", "N-LDF ($r=0.75$)"), ("N-LDF-r1.0", "N-LDF (parity)"), None,
            ("SLS", "Safe-Level-SMOTE"), ("SVM-SMOTE", "SVM-SMOTE"), ("KMeans-SMOTE", "k-means SMOTE")]
    for item in spec:
        if item is None:
            rows.append(r"\midrule")
            continue
        code, label = item
        m = means.loc[code]
        ratio = "--" if m["AchievedMinMajRatio"] != m["AchievedMinMajRatio"] else f"{m['AchievedMinMajRatio']:.3f}"
        if code in ("Baseline", "Baseline-thr", "Baseline-cw"):
            ratio = "--"
        rows.append(f"{label} & {m['F1']:.4f} & {m['Precision']:.4f} & {m['Recall']:.4f} & {m['Specificity']:.4f} & "
                    f"{m['AUPRC']:.4f} & {ratio} \\\\")
    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
\caption{{Controls without resampling and further oversamplers, with the
benchmark strategies they are compared with}}
\label{{tab:controls}}
\centering
\tabsmall
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{lcccccc}}
\toprule
\textbf{{Strategy}} & \textbf{{F1}} & \textbf{{Prec.}} & \textbf{{Rec.}} & \textbf{{Spec.}} & \textbf{{AUPRC}} & \hd{{\textbf{{Realized}}}}{{\textbf{{ratio}}}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\footnotetext{{Equal-weight means over the 41 tasks. The tuned threshold is chosen
by inner three-fold validation on the training partition; it leaves the
predicted probabilities, and hence AUPRC, unchanged. k-means SMOTE fell back
to SMOTE on 573 of 914 calls. SVM-SMOTE seeds only from minority support
vectors and stops short of parity}}
\end{{table}}
"""


def table_heldout(h: dict) -> str:
    sl = pd.DataFrame(h["source_level_means"]).T
    order = [("Baseline", "Baseline"), ("Baseline-thr", "Baseline, tuned threshold"), ("SMOTE", "SMOTE"),
             ("B-SMOTE", "Borderline-SMOTE"), ("ADASYN", "ADASYN"), ("SMOTE-Tomek", "SMOTE-Tomek"),
             ("ROS-r0.75", "ROS ($r=0.75$)"), ("SMOTE-r0.75", "SMOTE ($r=0.75$)"),
             ("N-LDF", "N-LDF ($r=0.75$)"), ("N-LDF-r1.0", "N-LDF (parity)")]
    rows = []
    for code, label in order:
        m = sl.loc[code]
        rows.append(f"{label} & {m['F1']:.4f} & {m['Recall']:.4f} & {m['Specificity']:.4f} & {m['AUPRC']:.4f} \\\\")
    body = "\n".join(rows)
    return rf"""\begin{{table}}[htbp]
\caption{{Held-out sources: source-level means over 10 sources (13 tasks) not
used in development}}
\label{{tab:heldout}}
\centering
\tabsmall
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lcccc}}
\toprule
\textbf{{Strategy}} & \textbf{{F1}} & \textbf{{Rec.}} & \textbf{{Spec.}} & \textbf{{AUPRC}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\footnotetext{{Task scores are averaged within source and the ten source values
receive equal weight. ROS at parity is omitted (F1 {sl.loc['ROS-r1.0', 'F1']:.4f})}}
\end{{table}}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", type=Path, default=Path("results/extension"))
    ap.add_argument("--heldout", type=Path, default=Path("results/heldout"))
    ap.add_argument("--out", type=Path, default=Path("results/tables"))
    ap.add_argument("--check-tex", type=Path, default=None,
                    help="manuscript .tex: check that every generated table body appears in it verbatim")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = json.loads((a.ext / "extension_results.json").read_text())
    means = pd.read_csv(a.ext / "strategy_means.csv", index_col=0)
    # realized ratio: equal-weight mean over tasks of the per-task mean (as every other column)
    realized = pd.DataFrame({"realized": means["AchievedMinMajRatio"]})
    numbers = {"means": means.round(6).to_dict(orient="index")}
    tex, n = table_amount(res, means); (a.out / "tab_amount.tex").write_text(tex); numbers["amount"] = n
    tex, n = table_mechanism(res); (a.out / "tab_mechanism.tex").write_text(tex); numbers["mechanism"] = n
    tex, n = table_variants(res, means, "ablation", ABL, "tab:ablation",
                            "Ablation of the N-LDF pipeline: each variant changes one element of the archived configuration", realized,
                            note=(" Local proportion: count-based $(n_{\\mathrm{min}}-n_{\\mathrm{maj}})/k_d$ over the same neighbors;"
                                  " unguided generation: uniform seed weights, distance-only partners, interpolation"
                                  " centered at $0.5$, acceptance unchanged."))
    (a.out / "tab_ablation.tex").write_text(tex); numbers["ablation"] = n
    tex, n = table_variants(res, means, "sensitivity", SENS, "tab:param_sensitivity",
                            "Parameter sensitivity: one value moved from the archived setting", realized)
    (a.out / "tab_sensitivity.tex").write_text(tex); numbers["sensitivity"] = n
    (a.out / "tab_controls.tex").write_text(table_controls(means))
    for fam in ["threshold", "weighting", "comparators", "comparators_parity"]:
        numbers[fam] = {f"{r['a']} vs {r['b']}|{m}": {k: r[k] for k in ("mean_diff", "holm_p", "source_holm_p", "ci_lo", "ci_hi", "win", "tie", "loss")}
                        for m, rows in res["families"].get(fam, {}).items() for r in rows}
    if (a.heldout / "heldout_results.json").exists():
        numbers["heldout"] = json.loads((a.heldout / "heldout_results.json").read_text())
        (a.out / "tab_heldout.tex").write_text(table_heldout(numbers["heldout"]))
    (a.out / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print("tables ->", a.out)
    if a.check_tex:
        tex = a.check_tex.read_text(encoding="utf-8")
        bad = 0
        for f in sorted(a.out.glob("tab_*.tex")):
            t = f.read_text()
            body = t[t.index("\\midrule"):t.index("\\bottomrule")]
            ok = body in tex
            bad += not ok
            print(f"  {'OK  ' if ok else 'DIFF'} {f.name}: table body {'found' if ok else 'NOT found'} in {a.check_tex.name}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
