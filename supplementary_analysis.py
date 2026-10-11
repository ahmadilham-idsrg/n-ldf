#!/usr/bin/env python3
"""Supplementary analyses computed from the archived fold-level records.

Nothing here retrains a model. Every number is derived from
``fold_level_results.csv`` (SHA-256 f0c957d5...7d6a4), the same file that
``reproduce_paper_tables.py`` reads.

Analyses (manuscript location in parentheses):

  A. Inference on the threshold-free metrics AUPRC and AUC-ROC, and on
     recall/specificity, with the same Friedman / Wilcoxon-Pratt / Holm
     procedure used for F1 (Section 6.2, "operating-point shift").
  B. Source-aware inference: (i) one block per source dataset (15 blocks),
     (ii) source-level cluster bootstrap of the mean paired difference,
     (iii) a linear mixed model with source and task random intercepts
     (Section 6.2 and Appendix B).
  C. Acceptance-test accounting by imbalance stratum (Section 5.2).
  D. Sensitivity: excluding the two chess tasks, and excluding all eleven
     tasks with categorical attributes (Appendix C).
  D3. Sensitivity to the aggregation rule: equal weight per classifier within
     each dataset-method block (hierarchical mean) instead of equal weight
     per completed fit (Appendix C).
  D4. Sensitivity to the ADASYN fallback: (a) the three tasks with any
     fallback call removed for every strategy; (b) only the fallback fits
     removed from ADASYN's block (shuttle-2_vs_5 then has no ADASYN block
     and drops out of the paired analysis) (Appendix C).
  E. Per-classifier F1 inference (Appendix D), with the source-aware
     counterparts (one block per source, Holm over 20; source-cluster
     bootstrap intervals) and the direct method x classifier interaction
     contrast between Random Forest and the linear SVM (Section 6.4).
  E3. The same interaction contrast over the full family of six classifier
     pairs x four oversamplers, Holm over 24 (sensitivity to the family).
  B3. Equivalence margins: 90% source-cluster bootstrap intervals of the
     oversampler differences (two one-sided tests at 5%) and the smallest
     symmetric margin at which equivalence would hold (Section 6.2).
  B2. Paired-difference models with source clustering: an intercept-only
     regression of each paired difference with cluster-robust standard
     errors (15 source clusters, t with 14 df), and a random-slope mixed
     model (source-specific N-LDF effect) as a check (Section 6.2, Appendix B).
  F. The 25 fit slots that raised exceptions, and the 31 ADASYN calls that
     fell back to SMOTE (Appendix A, Section 5.2).
  G. Source-cluster bootstrap intervals for the severity figure (Figure 4).

Usage
    python supplementary_analysis.py --input results/fold_level_results.csv
    python supplementary_analysis.py --check      # compare with manuscript
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METHODS = ["Baseline", "ADASYN", "B-SMOTE", "SMOTE", "SMOTE-Tomek", "N-LDF"]
OVERSAMPLERS = ["ADASYN", "B-SMOTE", "SMOTE", "SMOTE-Tomek", "N-LDF"]
STRATA = ["Mild", "High", "Extreme"]
METRICS = ["F1", "Precision", "Recall", "Specificity", "G-Mean", "AUC-ROC", "AUPRC"]
CHESS = ["kr-vs-k-zero_vs_eight", "kr-vs-k-zero_vs_fifteen"]
SEED = 42
B_BOOT = 10_000

# Values printed in the revised manuscript (Sections 5.2, 6.2, Appendices).
PAPER = {
    "auprc_friedman_all_p": 0.0027,
    "auprc_friedman_os_p": 5.04e-4,
    "auprc_nldf_vs_baseline_holm": 0.0625,
    "auprc_nldf_vs_smote_holm": 0.1258,
    "auprc_nldf_vs_bsmote_holm": 0.0138,
    "auprc_nldf_vs_adasyn_holm": 0.0749,
    "auprc_nldf_vs_tomek_holm": 0.1258,
    "auc_friedman_all_p": 4.09e-4,
    "auc_nldf_vs_baseline_holm": 0.0084,
    "auc_nldf_vs_smote_holm": 0.1407,
    "recall_baseline_mean": 0.4583,
    "specificity_baseline_mean": 0.9834,
    "source_block_nldf_vs_baseline_p": 0.0169,     # Holm over 5, 15 source blocks
    "mixed_nldf_vs_baseline_p": 1.78e-12,
    "mixed_min_oversampler_p": 0.2332,
    "factor_rejections_extreme_pct": 0.83,
    "distance_rejections_chess_share": 0.9959,
    "sens_chess_friedman_os_p": 0.6587,
    "sens_chess_nldf_f1": 0.5632,
    "failed_slots": 25,
    "failed_datasets": 17,
    "adasyn_fallback_calls": 31,
    "adasyn_fallback_fits": 124,
    "rf_vs_smote_holm20": 0.0263,
    "svml_vs_adasyn_holm20": 0.0144,
    # review-2 additions (Section 6.2, 6.4, Appendix B, D)
    "rf_vs_smote_source_holm20": 0.0921,
    "svml_vs_adasyn_source_holm20": 1.0,
    "interaction_smote_source_holm4": 0.0173,
    "interaction_smote_contrast": -0.0593,
    "interaction_global_lrt_p": 0.367,
    "clrobust_f1_baseline_p": 7.31e-5,
    "clrobust_f1_min_oversampler_p": 0.446,
    "clrobust_auprc_baseline_p": 0.0664,
    "random_slope_auprc_baseline_p": 0.407,
    "random_slope_auprc_adasyn_p": 0.0384,
    # review-3 additions (Section 6.2, 6.4, Appendix D)
    "interaction_smote_holm24": 0.1329,
    "interaction_adasyn_holm24": 0.0973,
    "interaction_tomek_holm24": 0.2793,
    "equiv_f1_smote_margin": 0.0225,
    "equiv_f1_adasyn_margin": 0.0402,
    "equiv_f1_max_noninferiority_margin": 0.0219,
    # review-5 additions (Appendix C)
    "sens_eqclf_max_change": 0.0052,
    "sens_eqclf_tomek_rank": 3.049,
    "sens_eqclf_nldf_rank": 3.134,
    "sens_eqclf_friedman_os_p": 0.7410,
    "sens_fallback_tasks_n": 38,
    "sens_fallback_nldf_f1": 0.5496,
    "sens_fallback_adasyn_f1": 0.5335,
    "sens_fallback_adasyn_holm": 1.0,
    "sens_fallback_friedman_os_p": 0.7528,
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def source_of(name: str) -> str:
    """Source dataset of a KEEL task, by name prefix (Section 5.1)."""
    for pfx in ["yeast", "abalone", "ecoli", "kddcup", "winequality", "glass",
                "poker", "kr-vs-k"]:
        if name.startswith(pfx):
            return pfx
    return name          # singleton tasks are their own source


def holm(rows: list[dict], key: str = "raw_p") -> list[dict]:
    rows = sorted(rows, key=lambda r: r[key])
    k, running = len(rows), 0.0
    for i, r in enumerate(rows):
        running = max(running, min(1.0, r[key] * (k - i)))
        r["holm_p"] = running
    return rows


def dataset_method_means(fold: pd.DataFrame) -> pd.DataFrame:
    return fold.groupby(["Dataset", "Method"])[METRICS].mean().reset_index()


def wide(dmm: pd.DataFrame, metric: str) -> pd.DataFrame:
    return dmm.pivot(index="Dataset", columns="Method", values=metric)[METHODS].dropna()


def wilcoxon(a, b) -> tuple[float, float]:
    w, p = stats.wilcoxon(a, b, zero_method="pratt", alternative="two-sided")
    return float(w), float(p)


# --------------------------------------------------------------------------
# A. inference per metric (task-level, as in Section 6.2)
# --------------------------------------------------------------------------
def inference(x: pd.DataFrame) -> dict:
    """x: tasks x methods. Friedman (6 and 5), planned family (N-LDF vs 5,
    Holm over 5), exploratory family (15 pairs, Holm over 15), paired summaries."""
    chi_all, p_all = stats.friedmanchisquare(*[x[m] for m in METHODS])
    chi_os, p_os = stats.friedmanchisquare(*[x[m] for m in OVERSAMPLERS])
    planned = []
    for m in METHODS:
        if m == "N-LDF":
            continue
        w, p = wilcoxon(x["N-LDF"], x[m])
        d = x["N-LDF"] - x[m]
        planned.append({"vs": m, "W": w, "raw_p": p, "mean_diff": float(d.mean()),
                        "median_diff": float(d.median()),
                        "win": int((d > 1e-9).sum()), "tie": int((d.abs() <= 1e-9).sum()),
                        "loss": int((d < -1e-9).sum())})
    planned = holm(planned)
    expl = []
    for a, b in itertools.combinations(METHODS, 2):
        w, p = wilcoxon(x[a], x[b])
        expl.append({"a": a, "b": b, "W": w, "raw_p": p,
                     "mean_diff": float((x[a] - x[b]).mean())})
    expl = holm(expl)
    means = x.mean().to_dict()
    rank = x[METHODS].rank(axis=1, ascending=False).mean().to_dict()   # six-strategy ranks, as in Table 5
    return {"n": len(x), "friedman_all": (float(chi_all), float(p_all)),
            "friedman_oversamplers": (float(chi_os), float(p_os)),
            "planned": planned, "exploratory": expl, "means": means,
            "mean_rank": rank}


# --------------------------------------------------------------------------
# B. source-aware inference
# --------------------------------------------------------------------------
def source_block_tests(x: pd.DataFrame) -> list[dict]:
    """Average each task-level paired difference within its source dataset and
    test the 15 source-level differences (Wilcoxon-Pratt, Holm over 5)."""
    src = pd.Series({t: source_of(t) for t in x.index})
    rows = []
    for m in METHODS:
        if m == "N-LDF":
            continue
        d = (x["N-LDF"] - x[m]).groupby(src).mean()
        w, p = wilcoxon(d, np.zeros(len(d)))
        rows.append({"vs": m, "n_sources": int(len(d)), "W": w, "raw_p": p,
                     "mean_diff": float(d.mean()),
                     "sources_positive": int((d > 1e-9).sum()),
                     "sources_negative": int((d < -1e-9).sum())})
    return holm(rows)


def cluster_bootstrap(x: pd.DataFrame, b: int = B_BOOT, seed: int = SEED) -> list[dict]:
    """Percentile interval for the mean paired difference N-LDF minus each
    method, resampling the 15 source datasets with replacement (all tasks of a
    drawn source enter together)."""
    rng = np.random.default_rng(seed)
    src = pd.Series({t: source_of(t) for t in x.index})
    groups = {s: x.index[src == s].tolist() for s in src.unique()}
    names = list(groups)
    rows = []
    for m in METHODS:
        if m == "N-LDF":
            continue
        d = (x["N-LDF"] - x[m])
        boots = np.empty(b)
        for i in range(b):
            pick = rng.choice(names, size=len(names), replace=True)
            tasks = list(itertools.chain.from_iterable(groups[s] for s in pick))
            boots[i] = d.loc[tasks].mean()
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"vs": m, "mean_diff": float(d.mean()), "ci_lo": float(lo),
                     "ci_hi": float(hi), "excludes_zero": bool(lo > 0 or hi < 0),
                     "p_two_sided": float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))})
    return rows


def mixed_model(dmm: pd.DataFrame, metric: str) -> dict:
    """y_tm = mu + beta_method + u_source + v_task + e, random intercepts for
    source and for task nested in source, fitted by REML (statsmodels MixedLM).
    Contrasts are each method minus N-LDF (treatment coding).

    Kept for the record (it was the third procedure of the first revision);
    the manuscript now reports cluster_robust_difference() and
    random_slope_model() instead, because the two random intercepts cancel in
    a within-task paired difference."""
    import statsmodels.formula.api as smf
    d = dmm[["Dataset", "Method", metric]].rename(columns={metric: "y"}).copy()
    d["Source"] = d.Dataset.map(source_of)
    d["Method"] = pd.Categorical(d.Method, categories=["N-LDF"] + [m for m in METHODS if m != "N-LDF"])
    model = smf.mixedlm("y ~ C(Method)", data=d, groups="Source",
                        re_formula="1", vc_formula={"Task": "0 + C(Dataset)"})
    fit = model.fit(reml=True, method=["lbfgs"])
    out = {}
    for m in METHODS:
        if m == "N-LDF":
            continue
        key = f"C(Method)[T.{m}]"
        out[m] = {"estimate_minus_nldf": float(fit.params[key]),
                  "se": float(fit.bse[key]), "p": float(fit.pvalues[key])}
    var_source = float(fit.cov_re.iloc[0, 0])
    var_task = float(fit.vcomp[0]) if len(fit.vcomp) else float("nan")
    return {"contrasts": out, "var_source": var_source, "var_task": var_task,
            "var_resid": float(fit.scale), "converged": bool(fit.converged)}


def cluster_robust_difference(x: pd.DataFrame) -> list[dict]:
    """Intercept-only regression of the paired difference N-LDF minus each
    method, with standard errors clustered by source (15 clusters) and a
    t reference distribution with G - 1 = 14 degrees of freedom. This is the
    model on paired differences with source grouping: the source and task
    intercepts of the score model cancel in a within-task difference, so the
    uncertainty of the contrast is taken from the between-source variation of
    the difference itself."""
    import statsmodels.api as sm
    src = np.array([source_of(t) for t in x.index])
    groups = pd.factorize(src)[0]
    rows = []
    for m in METHODS:
        if m == "N-LDF":
            continue
        d = (x["N-LDF"] - x[m]).values
        fit = sm.OLS(d, np.ones(len(d))).fit(cov_type="cluster", cov_kwds={"groups": groups})
        est, se = float(fit.params[0]), float(fit.bse[0])
        t = est / se
        rows.append({"vs": m, "estimate": est, "se_cluster": se, "t": float(t),
                     "df": int(len(set(groups)) - 1),
                     "raw_p": float(2 * stats.t.sf(abs(t), df=len(set(groups)) - 1))})
    return holm(rows)


def random_slope_model(dmm: pd.DataFrame, metric: str) -> dict:
    """y_tm = mu + beta_method + u_source + w_source * [m = N-LDF] + v_task + e:
    the score model with an additional source-specific N-LDF effect (random
    slope on the N-LDF indicator), the parsimonious alternative to a full
    random-slope structure with only 15 sources. REML, statsmodels MixedLM."""
    import statsmodels.formula.api as smf
    d = dmm[["Dataset", "Method", metric]].rename(columns={metric: "y"}).copy()
    d["Source"] = d.Dataset.map(source_of)
    d["nldf"] = (d.Method == "N-LDF").astype(float)
    d["Method"] = pd.Categorical(d.Method, categories=["N-LDF"] + [m for m in METHODS if m != "N-LDF"])
    model = smf.mixedlm("y ~ C(Method)", data=d, groups="Source", re_formula="1 + nldf",
                        vc_formula={"Task": "0 + C(Dataset)"})
    fit = model.fit(reml=True, method=["lbfgs", "powell"])
    out = {}
    for m in METHODS:
        if m == "N-LDF":
            continue
        key = f"C(Method)[T.{m}]"
        out[m] = {"estimate_minus_nldf": float(fit.params[key]),
                  "se": float(fit.bse[key]), "p": float(fit.pvalues[key])}
    return {"contrasts": out, "var_source_intercept": float(fit.cov_re.iloc[0, 0]),
            "var_source_nldf_slope": float(fit.cov_re.iloc[1, 1]),
            "var_task": float(fit.vcomp[0]) if len(fit.vcomp) else float("nan"),
            "var_resid": float(fit.scale), "converged": bool(fit.converged)}


def per_classifier_source_aware(fold: pd.DataFrame, b: int = B_BOOT, seed: int = SEED) -> dict:
    """For each classifier: the planned family on that classifier's
    dataset-level F1 with (i) one block per source (Wilcoxon-Pratt on the 15
    source-level mean differences) and (ii) a source-cluster bootstrap
    interval. Holm is applied over all 20 source-level comparisons."""
    per = fold.groupby(["Classifier", "Dataset", "Method"])["F1"].mean().reset_index()
    all20, boots = [], []
    for clf, g in per.groupby("Classifier"):
        x = g.pivot(index="Dataset", columns="Method", values="F1")[METHODS].dropna()
        for r in source_block_tests(x):
            all20.append({"classifier": clf, "vs": r["vs"], "raw_p": r["raw_p"],
                          "mean_diff": r["mean_diff"], "sources_positive": r["sources_positive"],
                          "sources_negative": r["sources_negative"]})
        for r in cluster_bootstrap(x, b=b, seed=seed):
            r["classifier"] = clf
            boots.append(r)
    return {"source_block_holm20": holm(all20), "cluster_bootstrap": boots}


def interaction_contrasts(fold: pd.DataFrame, clf_a: str = "RF", clf_b: str = "SVM-Linear",
                          b: int = B_BOOT, seed: int = SEED) -> dict:
    """Direct method x classifier contrast: for each comparison strategy m,
    D_t = (F1[N-LDF] - F1[m] | clf_a) - (F1[N-LDF] - F1[m] | clf_b) on task t.
    Tested with one block per source (Wilcoxon-Pratt, Holm over the four
    oversampler contrasts) and a source-cluster bootstrap interval. The pair
    (clf_a, clf_b) was chosen after inspecting Table 9, so the contrast is
    exploratory. A global likelihood-ratio test of the method x classifier
    interaction (ML fits of the score model with and without the interaction,
    source and task random intercepts) is reported alongside."""
    import statsmodels.formula.api as smf
    dcm = fold.groupby(["Dataset", "Classifier", "Method"], as_index=False)["F1"].mean()
    piv = dcm.pivot_table(index="Dataset", columns=["Classifier", "Method"], values="F1")
    src = pd.Series({t: source_of(t) for t in piv.index})
    groups = {s: piv.index[src == s].tolist() for s in src.unique()}
    names = list(groups)
    rng = np.random.default_rng(seed)
    rows = []
    for m in METHODS:
        if m == "N-LDF":
            continue
        D = (piv[(clf_a, "N-LDF")] - piv[(clf_a, m)]) - (piv[(clf_b, "N-LDF")] - piv[(clf_b, m)])
        d_src = D.groupby(src).mean()
        _, p_src = wilcoxon(d_src, np.zeros(len(d_src)))
        _, p_task = wilcoxon(D, np.zeros(len(D)))
        bs = np.empty(b)
        for i in range(b):
            pick = rng.choice(names, size=len(names), replace=True)
            tasks = list(itertools.chain.from_iterable(groups[s] for s in pick))
            bs[i] = D.loc[tasks].mean()
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append({"vs": m, "mean_contrast": float(D.mean()), "task_p": float(p_task),
                     "raw_p": float(p_src), "ci_lo": float(lo), "ci_hi": float(hi),
                     "sources_negative": int((d_src < -1e-9).sum()), "n_sources": int(len(d_src))})
    os_rows = holm([r for r in rows if r["vs"] != "Baseline"])
    base = [r for r in rows if r["vs"] == "Baseline"]
    for r in base:
        r["holm_p"] = float("nan")
    d = dcm.rename(columns={"F1": "y"}).copy()
    d["Source"] = d.Dataset.map(source_of)
    d["Method"] = pd.Categorical(d.Method, categories=["N-LDF"] + [m for m in METHODS if m != "N-LDF"])
    kw = dict(groups="Source", re_formula="1", vc_formula={"Task": "0 + C(Dataset)"})
    full = smf.mixedlm("y ~ C(Method)*C(Classifier)", data=d, **kw).fit(reml=False, method=["lbfgs"])
    red = smf.mixedlm("y ~ C(Method)+C(Classifier)", data=d, **kw).fit(reml=False, method=["lbfgs"])
    lr = float(2 * (full.llf - red.llf))
    df = int(len(full.params) - len(red.params))
    return {"clf_a": clf_a, "clf_b": clf_b, "contrasts": os_rows + base,
            "global_lrt": {"LR": lr, "df": df, "p": float(stats.chi2.sf(lr, df)),
                           "converged": bool(full.converged and red.converged)}}


def interaction_all_pairs(fold: pd.DataFrame, classifiers=("RF", "SVM-Linear", "SVM-RBF", "XGBoost")) -> list[dict]:
    """The interaction contrast of interaction_contrasts() for every pair of
    classifiers and every oversampler (6 x 4 = 24 contrasts), one block per
    source, Holm over all 24. Sensitivity of the RF-vs-linear-SVM result to the
    choice of contrast family."""
    dcm = fold.groupby(["Dataset", "Classifier", "Method"], as_index=False)["F1"].mean()
    piv = dcm.pivot_table(index="Dataset", columns=["Classifier", "Method"], values="F1")
    src = pd.Series({t: source_of(t) for t in piv.index})
    rows = []
    for a, b in itertools.combinations(classifiers, 2):
        for m in [mm for mm in OVERSAMPLERS if mm != "N-LDF"]:
            D = (piv[(a, "N-LDF")] - piv[(a, m)]) - (piv[(b, "N-LDF")] - piv[(b, m)])
            d_src = D.groupby(src).mean()
            _, p = wilcoxon(d_src, np.zeros(len(d_src)))
            rows.append({"pair": f"{a} vs {b}", "vs": m, "mean_contrast": float(D.mean()), "raw_p": float(p)})
    return holm(rows)


def equivalence_margins(x: pd.DataFrame, b: int = B_BOOT, seed: int = SEED, level: float = 0.90) -> list[dict]:
    """Two one-sided tests at 5% via the 90% source-cluster bootstrap interval
    of the paired difference N-LDF minus each oversampler: equivalence within
    +/- delta holds iff the interval lies inside [-delta, +delta]. Reports the
    smallest such delta (max(|lo|, |hi|)) and the non-inferiority margin
    (-lo if lo < 0 else 0). No margin was fixed in advance; these are the
    margins at which the claims *would* hold."""
    rng = np.random.default_rng(seed)
    src = pd.Series({t: source_of(t) for t in x.index})
    groups = {s_: x.index[src == s_].tolist() for s_ in src.unique()}
    names = list(groups)
    lo_q, hi_q = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    rows = []
    for m in OVERSAMPLERS:
        if m == "N-LDF":
            continue
        d = x["N-LDF"] - x[m]
        bs = np.empty(b)
        for i in range(b):
            pick = rng.choice(names, size=len(names), replace=True)
            bs[i] = d.loc[list(itertools.chain.from_iterable(groups[s_] for s_ in pick))].mean()
        lo, hi = np.percentile(bs, [lo_q, hi_q])
        rows.append({"vs": m, "mean_diff": float(d.mean()), "ci90_lo": float(lo), "ci90_hi": float(hi),
                     "smallest_equivalence_margin": float(max(abs(lo), abs(hi))),
                     "noninferiority_margin": float(max(0.0, -lo))})
    return rows


# --------------------------------------------------------------------------
# C. acceptance accounting by stratum
# --------------------------------------------------------------------------
def rejections_by_stratum(fold: pd.DataFrame) -> pd.DataFrame:
    calls = (fold[fold.Method == "N-LDF"]
                 .drop_duplicates(["Dataset", "Repeat", "Fold"]).copy())
    calls["none"] = (calls.n_attempted - calls.n_accepted - calls.n_rejected_distance
                     - calls.n_rejected_maj - calls.n_rejected_density)
    g = calls.groupby("Severity")[["n_attempted", "n_accepted", "n_rejected_distance",
                                   "n_rejected_maj", "n_rejected_density", "none"]].sum()
    g = g.reindex(STRATA)
    g.loc["All"] = g.sum()
    g = g.astype(int)
    for c in ["n_rejected_distance", "n_rejected_maj", "n_rejected_density"]:
        g[c + "_pct"] = 100 * g[c] / g.n_attempted
    chess = calls[calls.Dataset.isin(CHESS)]
    g.attrs["chess_distance_share"] = float(chess.n_rejected_distance.sum()
                                            / calls.n_rejected_distance.sum())
    g.attrs["chess_distance_rejections"] = int(chess.n_rejected_distance.sum())
    g.attrs["chess_attempts"] = int(chess.n_attempted.sum())
    g.attrs["chess_accepted"] = int(chess.n_accepted.sum())
    g.attrs["calls_per_stratum"] = calls.groupby("Severity").size().reindex(STRATA).astype(int).to_dict()
    # the kappa threshold implied by eta >= 0 at the stratum's mean fold IR
    ir = calls.groupby("Severity").IR.mean().reindex(STRATA)
    g.attrs["kappa_threshold"] = {s: float(ir[s] ** -0.7) for s in STRATA}
    g.attrs["mean_fold_ir"] = {s: float(ir[s]) for s in STRATA}
    return g


# --------------------------------------------------------------------------
# F. fit slots and sampler fallbacks
# --------------------------------------------------------------------------
def failed_slots(fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in fold.groupby("Dataset"):
        reps, folds = sorted(g.Repeat.unique()), sorted(g.Fold.unique())
        have = set(map(tuple, g[["Method", "Classifier", "Repeat", "Fold"]].to_numpy()))
        for m, c, r, f in itertools.product(METHODS, sorted(fold.Classifier.unique()), reps, folds):
            if (m, c, r, f) not in have:
                rows.append({"Dataset": ds, "Method": m, "Classifier": c,
                             "Repeat": int(r), "Fold": int(f)})
    return pd.DataFrame(rows)


def adasyn_fallbacks(fold: pd.DataFrame) -> dict:
    fb = fold[fold.SamplerStatus == "fallback"]
    calls = fb.drop_duplicates(["Dataset", "Repeat", "Fold"])
    return {"fits": int(len(fb)), "calls": int(len(calls)),
            "methods": fb.Method.unique().tolist(),
            "fallback_to": fb.SamplerFallback.unique().tolist(),
            "by_dataset": calls.groupby("Dataset").size().to_dict(),
            "message": fb.SamplerError.iloc[0] if len(fb) else ""}


# --------------------------------------------------------------------------
# G. severity means with source-cluster bootstrap intervals
# --------------------------------------------------------------------------
def severity_cluster_bootstrap(dmm: pd.DataFrame, fold: pd.DataFrame,
                               b: int = B_BOOT, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sev = fold[["Dataset", "Severity"]].drop_duplicates("Dataset").set_index("Dataset").Severity
    f1 = wide(dmm, "F1")
    rows = []
    for s in STRATA:
        tasks = f1.index[sev.loc[f1.index] == s]
        src = pd.Series({t: source_of(t) for t in tasks})
        groups = {k: src.index[src == k].tolist() for k in src.unique()}
        names = list(groups)
        for m in METHODS:
            v = f1.loc[tasks, m]
            boots = np.empty(b)
            for i in range(b):
                pick = rng.choice(names, size=len(names), replace=True)
                boots[i] = v.loc[list(itertools.chain.from_iterable(groups[k] for k in pick))].mean()
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"Severity": s, "Method": m, "n_tasks": int(len(tasks)),
                         "n_sources": len(names), "mean": float(v.mean()),
                         "ci_lo": float(lo), "ci_hi": float(hi)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def fmt_p(p: float) -> str:
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="results/fold_level_results.csv", type=Path)
    ap.add_argument("--outdir", default="results/supplementary", type=Path)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--boot", type=int, default=B_BOOT)
    args = ap.parse_args()
    if not args.input.exists():
        print(f"error: {args.input} not found", file=sys.stderr)
        return 2
    args.outdir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 140)

    fold = pd.read_csv(args.input)
    dmm = dataset_method_means(fold)
    results: dict = {}

    # ---- A -------------------------------------------------------------
    print("=" * 78)
    print("A. Task-level inference per metric (Friedman; Wilcoxon-Pratt; Holm)")
    print("=" * 78)
    for metric in ["F1", "AUPRC", "AUC-ROC", "Recall", "Specificity", "G-Mean", "Precision"]:
        inf = inference(wide(dmm, metric))
        results[f"inference_{metric}"] = inf
        print(f"\n[{metric}]  means: " + ", ".join(f"{m} {inf['means'][m]:.4f}" for m in METHODS))
        print(f"  Friedman all six : chi2={inf['friedman_all'][0]:.4f} p={fmt_p(inf['friedman_all'][1])}")
        print(f"  Friedman five OS : chi2={inf['friedman_oversamplers'][0]:.4f} p={fmt_p(inf['friedman_oversamplers'][1])}")
        print("  planned family (N-LDF vs ...):")
        for r in sorted(inf["planned"], key=lambda r: METHODS.index(r["vs"])):
            print(f"    {r['vs']:<12} W={r['W']:7.1f} raw={fmt_p(r['raw_p'])} Holm={fmt_p(r['holm_p'])}"
                  f"  mean diff={r['mean_diff']:+.4f} W-T-L={r['win']}-{r['tie']}-{r['loss']}")
        print("  mean rank (six strategies): " + ", ".join(f"{m} {inf['mean_rank'][m]:.3f}" for m in METHODS))
        bl = [r for r in inf["exploratory"] if "Baseline" in (r["a"], r["b"])]
        os_ = [r for r in inf["exploratory"] if "Baseline" not in (r["a"], r["b"])]
        print(f"  exploratory family: baseline pairs Holm min={min(r['holm_p'] for r in bl):.4f} "
              f"max={max(r['holm_p'] for r in bl):.4f}; oversampler pairs Holm min={min(r['holm_p'] for r in os_):.4f}")
        sig = [r for r in inf["exploratory"] if r["holm_p"] < 0.05]
        if sig:
            print("  exploratory pairs significant after Holm-15: " +
                  "; ".join(f"{r['a']} vs {r['b']} (Holm {r['holm_p']:.4f}, diff {r['mean_diff']:+.4f})" for r in sig))

    # ---- B -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("B. Source-aware inference (15 source datasets)")
    print("=" * 78)
    for metric in ["F1", "AUPRC"]:
        x = wide(dmm, metric)
        sb = source_block_tests(x)
        cb = cluster_bootstrap(x, b=args.boot)
        mm = mixed_model(dmm, metric)
        results[f"source_block_{metric}"] = sb
        results[f"cluster_boot_{metric}"] = cb
        results[f"mixed_{metric}"] = mm
        print(f"\n[{metric}] one block per source (Wilcoxon-Pratt on 15 source means, Holm over 5):")
        for r in sorted(sb, key=lambda r: METHODS.index(r["vs"])):
            print(f"    {r['vs']:<12} raw={fmt_p(r['raw_p'])} Holm={fmt_p(r['holm_p'])} "
                  f"mean={r['mean_diff']:+.4f} sources +/-={r['sources_positive']}/{r['sources_negative']}")
        print(f"[{metric}] source-cluster bootstrap ({args.boot} resamples) of the mean paired difference:")
        for r in sorted(cb, key=lambda r: METHODS.index(r["vs"])):
            print(f"    {r['vs']:<12} mean={r['mean_diff']:+.4f} 95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
                  f"  excludes 0: {r['excludes_zero']}")
        print(f"[{metric}] mixed model (source + task random intercepts; converged={mm['converged']}): "
              f"var source={mm['var_source']:.5f} task={mm['var_task']:.5f} resid={mm['var_resid']:.5f}")
        for m, r in mm["contrasts"].items():
            print(f"    {m:<12} minus N-LDF = {r['estimate_minus_nldf']:+.4f} (se {r['se']:.4f}) p={fmt_p(r['p'])}")

    # ---- C -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("C. Acceptance-test accounting by stratum (N-LDF calls)")
    print("=" * 78)
    rej = rejections_by_stratum(fold)
    results["rejections"] = {"table": rej.reset_index().to_dict(orient="records"), **rej.attrs}
    print(rej.to_string())
    print(f"  share of distance rejections from the two chess tasks: {rej.attrs['chess_distance_share']:.4f} "
          f"({rej.attrs['chess_distance_rejections']} of {int(rej.loc['All', 'n_rejected_distance'])}; "
          f"chess attempts {rej.attrs['chess_attempts']}, accepted {rej.attrs['chess_accepted']})")
    print(f"  calls per stratum: {rej.attrs['calls_per_stratum']}")
    print(f"  mean fold IR per stratum: {rej.attrs['mean_fold_ir']}")
    print(f"  kappa threshold IR^-0.7 for eta>=0: {rej.attrs['kappa_threshold']}")
    rej.to_csv(args.outdir / "rejections_by_stratum.csv")

    # ---- D -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("D. Sensitivity analyses (F1)")
    print("=" * 78)
    audit_cat = fold[["Dataset"]].drop_duplicates()
    cat_tasks = [d for d in fold.Dataset.unique()
                 if d.startswith(("abalone", "kddcup", "kr-vs-k"))]
    for label, excl in [("without the two chess tasks", CHESS),
                        ("without the eleven categorical tasks", cat_tasks)]:
        x = wide(dmm[~dmm.Dataset.isin(excl)], "F1")
        inf = inference(x)
        results[f"sensitivity_{label}"] = inf
        print(f"\n[{label}] n={inf['n']}  means: " + ", ".join(f"{m} {inf['means'][m]:.4f}" for m in METHODS))
        print(f"  mean rank (six strategies): " + ", ".join(f"{m} {inf['mean_rank'][m]:.3f}" for m in METHODS))
        print(f"  Friedman all={fmt_p(inf['friedman_all'][1])} five={fmt_p(inf['friedman_oversamplers'][1])}")
        for r in sorted(inf["planned"], key=lambda r: METHODS.index(r["vs"])):
            print(f"    vs {r['vs']:<12} Holm={fmt_p(r['holm_p'])} mean diff={r['mean_diff']:+.4f} W-T-L={r['win']}-{r['tie']}-{r['loss']}")

    # ---- D3: equal-classifier aggregation ------------------------------
    hier = (fold.groupby(["Dataset", "Method", "Classifier"])["F1"].mean()
                .groupby(["Dataset", "Method"]).mean().reset_index())
    x = hier.pivot(index="Dataset", columns="Method", values="F1")[METHODS].dropna()
    inf = inference(x)
    results["sensitivity_equal_classifier"] = inf
    print(f"\n[equal weight per classifier] n={inf['n']}  means: " + ", ".join(f"{m} {inf['means'][m]:.4f}" for m in METHODS))
    print(f"  mean rank (six strategies): " + ", ".join(f"{m} {inf['mean_rank'][m]:.3f}" for m in METHODS))
    print(f"  Friedman all={fmt_p(inf['friedman_all'][1])} five={fmt_p(inf['friedman_oversamplers'][1])}")
    for r in sorted(inf["planned"], key=lambda r: METHODS.index(r["vs"])):
        print(f"    vs {r['vs']:<12} Holm={fmt_p(r['holm_p'])} mean diff={r['mean_diff']:+.4f} W-T-L={r['win']}-{r['tie']}-{r['loss']}")
    # largest change in a dataset-level mean between the two aggregation rules
    dd = (hier.set_index(["Dataset", "Method"]).F1 - dmm.set_index(["Dataset", "Method"]).F1).abs()
    results["sensitivity_equal_classifier"]["max_abs_change_dataset_mean"] = float(dd.max())
    print(f"  largest |change| in a dataset-method mean: {dd.max():.4f} ({dd.idxmax()})")

    # ---- D4: ADASYN fallback ------------------------------------------
    fb_tasks = sorted(fold[(fold.Method == "ADASYN") & (fold.SamplerStatus == "fallback")].Dataset.unique())
    results["adasyn_fallback_tasks"] = fb_tasks
    x = wide(dmm[~dmm.Dataset.isin(fb_tasks)], "F1")
    inf = inference(x)
    results["sensitivity_without_fallback_tasks"] = inf
    print(f"\n[without the {len(fb_tasks)} tasks with an ADASYN fallback: {fb_tasks}] n={inf['n']}  means: "
          + ", ".join(f"{m} {inf['means'][m]:.4f}" for m in METHODS))
    print(f"  mean rank (six strategies): " + ", ".join(f"{m} {inf['mean_rank'][m]:.3f}" for m in METHODS))
    print(f"  Friedman all={fmt_p(inf['friedman_all'][1])} five={fmt_p(inf['friedman_oversamplers'][1])}")
    for r in sorted(inf["planned"], key=lambda r: METHODS.index(r["vs"])):
        print(f"    vs {r['vs']:<12} Holm={fmt_p(r['holm_p'])} mean diff={r['mean_diff']:+.4f} W-T-L={r['win']}-{r['tie']}-{r['loss']}")
    nofb = fold[~((fold.Method == "ADASYN") & (fold.SamplerStatus == "fallback"))]
    x = wide(dataset_method_means(nofb), "F1")
    inf = inference(x)
    results["sensitivity_without_fallback_fits"] = inf
    print(f"\n[ADASYN fallback fits removed from ADASYN's blocks] n={inf['n']}  means: "
          + ", ".join(f"{m} {inf['means'][m]:.4f}" for m in METHODS))
    for r in sorted(inf["planned"], key=lambda r: METHODS.index(r["vs"])):
        print(f"    vs {r['vs']:<12} Holm={fmt_p(r['holm_p'])} mean diff={r['mean_diff']:+.4f} W-T-L={r['win']}-{r['tie']}-{r['loss']}")

    # ---- E -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("E. Per-classifier F1 inference")
    print("=" * 78)
    per = fold.groupby(["Classifier", "Dataset", "Method"])["F1"].mean().reset_index()
    results["per_classifier"] = {}
    all20 = []
    for clf, g in per.groupby("Classifier"):
        x = g.pivot(index="Dataset", columns="Method", values="F1")[METHODS].dropna()
        inf = inference(x)
        results["per_classifier"][clf] = inf
        for r in inf["planned"]:
            all20.append({"classifier": clf, "vs": r["vs"], "raw_p": r["raw_p"],
                          "mean_diff": r["mean_diff"], "wtl": f"{r['win']}-{r['tie']}-{r['loss']}"})
        print(f"\n[{clf}] n={inf['n']} Friedman five OS p={fmt_p(inf['friedman_oversamplers'][1])}; "
              + ", ".join(f"{m} {inf['means'][m]:.4f}" for m in METHODS))
        for r in sorted(inf["planned"], key=lambda r: METHODS.index(r["vs"])):
            print(f"    vs {r['vs']:<12} raw={fmt_p(r['raw_p'])} Holm-5={fmt_p(r['holm_p'])} mean diff={r['mean_diff']:+.4f} W-T-L={r['win']}-{r['tie']}-{r['loss']}")
    all20 = holm(all20)
    results["per_classifier_holm20"] = all20
    print("\n  Holm over all 20 classifier-specific comparisons (N-LDF vs ...):")
    for r in sorted(all20, key=lambda r: (r["classifier"], METHODS.index(r["vs"]))):
        print(f"    {r['classifier']:<11} vs {r['vs']:<12} raw={fmt_p(r['raw_p'])} Holm-20={fmt_p(r['holm_p'])} "
              f"diff={r['mean_diff']:+.4f} W-T-L={r['wtl']}")

    # ---- E2: source-aware per classifier; interaction ------------------
    print("\n" + "=" * 78)
    print("E2. Per-classifier source-aware inference and the RF vs linear-SVM interaction")
    print("=" * 78)
    pcs = per_classifier_source_aware(fold, b=args.boot)
    results["per_classifier_source_block_holm20"] = pcs["source_block_holm20"]
    results["per_classifier_cluster_bootstrap"] = pcs["cluster_bootstrap"]
    print("  one block per source, Holm over 20 (N-LDF vs ...):")
    for r in sorted(pcs["source_block_holm20"], key=lambda r: (r["classifier"], METHODS.index(r["vs"]))):
        print(f"    {r['classifier']:<11} vs {r['vs']:<12} raw={fmt_p(r['raw_p'])} Holm-20={fmt_p(r['holm_p'])} "
              f"diff={r['mean_diff']:+.4f} sources +/-={r['sources_positive']}/{r['sources_negative']}")
    print("  source-cluster bootstrap 95% intervals:")
    for r in sorted(pcs["cluster_bootstrap"], key=lambda r: (r["classifier"], METHODS.index(r["vs"]))):
        print(f"    {r['classifier']:<11} vs {r['vs']:<12} diff={r['mean_diff']:+.4f} CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
              f"{'  excludes 0' if r['excludes_zero'] else ''}")
    inter = interaction_contrasts(fold, b=args.boot)
    results["interaction_rf_vs_svml"] = inter
    print(f"  interaction contrast (N-LDF - m | RF) - (N-LDF - m | SVM-Linear):")
    for r in inter["contrasts"]:
        print(f"    vs {r['vs']:<12} contrast={r['mean_contrast']:+.4f} source-p={fmt_p(r['raw_p'])} "
              f"Holm-4={fmt_p(r['holm_p'])} CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] sources negative={r['sources_negative']}/{r['n_sources']}")
    g = inter["global_lrt"]
    print(f"  global method x classifier LRT: LR={g['LR']:.2f} df={g['df']} p={fmt_p(g['p'])}")

    # ---- E3: the contrast over all six classifier pairs (Holm over 24) ---
    all24 = interaction_all_pairs(fold)
    results["interaction_all_pairs_holm24"] = all24
    print("  interaction contrasts over all six classifier pairs (one block per source, Holm over 24):")
    for r in sorted(all24, key=lambda r: (r["pair"], r["vs"])):
        if r["pair"] == "RF vs SVM-Linear" or r["holm_p"] < 1:
            print(f"    {r['pair']:<24} vs {r['vs']:<12} contrast={r['mean_contrast']:+.4f} raw={fmt_p(r['raw_p'])} Holm-24={fmt_p(r['holm_p'])}")

    # ---- B2: paired-difference models with source clustering ----------
    print("\n" + "=" * 78)
    print("B2. Paired-difference models with source clustering (F1, AUPRC)")
    print("=" * 78)
    for metric in ["F1", "AUPRC"]:
        cr = cluster_robust_difference(wide(dmm, metric))
        results[f"cluster_robust_{metric}"] = cr
        print(f"  [{metric}] cluster-robust regression of the paired difference (t, 14 df):")
        for r in sorted(cr, key=lambda r: METHODS.index(r["vs"])):
            print(f"    vs {r['vs']:<12} delta={r['estimate']:+.4f} SE={r['se_cluster']:.4f} t={r['t']:+.2f} raw p={fmt_p(r['raw_p'])} Holm-5={fmt_p(r['holm_p'])}")
        try:
            rs = random_slope_model(dmm, metric)
            results[f"random_slope_{metric}"] = rs
            print(f"  [{metric}] random-slope model (source-specific N-LDF effect; converged={rs['converged']}; "
                  f"var slope={rs['var_source_nldf_slope']:.6f}):")
            for m, c in rs["contrasts"].items():
                print(f"    {m:<12} m - N-LDF = {c['estimate_minus_nldf']:+.4f} (se {c['se']:.4f}) p={fmt_p(c['p'])}")
        except Exception as exc:  # pragma: no cover
            print(f"  [{metric}] random-slope model failed: {exc}")

    # ---- B3: equivalence margins (TOST via 90% source-cluster intervals) --
    print("\n" + "=" * 78)
    print("B3. Equivalence margins: 90% source-cluster bootstrap intervals (TOST at 5%)")
    print("=" * 78)
    for metric in ["F1", "AUPRC"]:
        eq = equivalence_margins(wide(dmm, metric), b=args.boot)
        results[f"equivalence_{metric}"] = eq
        for r in eq:
            print(f"  [{metric}] N-LDF - {r['vs']:<12} mean={r['mean_diff']:+.4f} 90% CI=[{r['ci90_lo']:+.4f}, {r['ci90_hi']:+.4f}] "
                  f"smallest equivalence margin={r['smallest_equivalence_margin']:.4f} non-inferiority margin={r['noninferiority_margin']:.4f}")

    # ---- F -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("F. Fit slots that raised exceptions; sampler fallbacks")
    print("=" * 78)
    miss = failed_slots(fold)
    miss.to_csv(args.outdir / "failed_fit_slots.csv", index=False)
    results["failed_slots"] = {"n": int(len(miss)), "datasets": int(miss.Dataset.nunique()),
                               "by_method": miss.Method.value_counts().to_dict(),
                               "by_classifier": miss.Classifier.value_counts().to_dict()}
    print(f"  missing slots: {len(miss)} across {miss.Dataset.nunique()} datasets")
    print(miss.sort_values(["Dataset", "Method", "Classifier", "Repeat", "Fold"]).to_string(index=False))
    fb = adasyn_fallbacks(fold)
    results["adasyn_fallbacks"] = fb
    print(f"\n  sampler fallbacks: {fb['fits']} fits = {fb['calls']} resampling calls, methods {fb['methods']},"
          f" to {fb['fallback_to']}; by dataset {fb['by_dataset']}")
    print(f"  library message: {fb['message']}")

    # ---- G -------------------------------------------------------------
    sevb = severity_cluster_bootstrap(dmm, fold, b=args.boot)
    sevb.to_csv(args.outdir / "fig_f1_severity_cluster_bootstrap.csv", index=False)
    results["severity_cluster_bootstrap"] = sevb.to_dict(orient="records")
    print("\n" + "=" * 78)
    print("G. Severity means with source-cluster bootstrap 95% intervals (Figure 4)")
    print("=" * 78)
    print(sevb.to_string(index=False))

    with open(args.outdir / "supplementary_results.json", "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    print(f"\nwritten: {args.outdir}")

    # ---- check ---------------------------------------------------------
    if args.check:
        def pl(metric, vs):
            return next(r for r in results[f"inference_{metric}"]["planned"] if r["vs"] == vs)["holm_p"]
        got = {
            "auprc_friedman_all_p": results["inference_AUPRC"]["friedman_all"][1],
            "auprc_friedman_os_p": results["inference_AUPRC"]["friedman_oversamplers"][1],
            "auprc_nldf_vs_baseline_holm": pl("AUPRC", "Baseline"),
            "auprc_nldf_vs_smote_holm": pl("AUPRC", "SMOTE"),
            "auprc_nldf_vs_bsmote_holm": pl("AUPRC", "B-SMOTE"),
            "auprc_nldf_vs_adasyn_holm": pl("AUPRC", "ADASYN"),
            "auprc_nldf_vs_tomek_holm": pl("AUPRC", "SMOTE-Tomek"),
            "auc_friedman_all_p": results["inference_AUC-ROC"]["friedman_all"][1],
            "auc_nldf_vs_baseline_holm": pl("AUC-ROC", "Baseline"),
            "auc_nldf_vs_smote_holm": pl("AUC-ROC", "SMOTE"),
            "recall_baseline_mean": results["inference_Recall"]["means"]["Baseline"],
            "specificity_baseline_mean": results["inference_Specificity"]["means"]["Baseline"],
            "source_block_nldf_vs_baseline_p": next(r for r in results["source_block_F1"] if r["vs"] == "Baseline")["holm_p"],
            "mixed_nldf_vs_baseline_p": results["mixed_F1"]["contrasts"]["Baseline"]["p"],
            "mixed_min_oversampler_p": min(results["mixed_F1"]["contrasts"][m]["p"] for m in OVERSAMPLERS if m != "N-LDF"),
            "factor_rejections_extreme_pct": float(rej.loc["Extreme", "n_rejected_density_pct"]),
            "distance_rejections_chess_share": rej.attrs["chess_distance_share"],
            "sens_chess_friedman_os_p": results["sensitivity_without the two chess tasks"]["friedman_oversamplers"][1],
            "sens_chess_nldf_f1": results["sensitivity_without the two chess tasks"]["means"]["N-LDF"],
            "sens_eqclf_max_change": results["sensitivity_equal_classifier"]["max_abs_change_dataset_mean"],
            "sens_eqclf_tomek_rank": results["sensitivity_equal_classifier"]["mean_rank"]["SMOTE-Tomek"],
            "sens_eqclf_nldf_rank": results["sensitivity_equal_classifier"]["mean_rank"]["N-LDF"],
            "sens_eqclf_friedman_os_p": results["sensitivity_equal_classifier"]["friedman_oversamplers"][1],
            "sens_fallback_tasks_n": results["sensitivity_without_fallback_tasks"]["n"],
            "sens_fallback_nldf_f1": results["sensitivity_without_fallback_tasks"]["means"]["N-LDF"],
            "sens_fallback_adasyn_f1": results["sensitivity_without_fallback_tasks"]["means"]["ADASYN"],
            "sens_fallback_adasyn_holm": next(r for r in results["sensitivity_without_fallback_tasks"]["planned"] if r["vs"] == "ADASYN")["holm_p"],
            "sens_fallback_friedman_os_p": results["sensitivity_without_fallback_tasks"]["friedman_oversamplers"][1],
            "failed_slots": results["failed_slots"]["n"],
            "failed_datasets": results["failed_slots"]["datasets"],
            "adasyn_fallback_calls": fb["calls"],
            "adasyn_fallback_fits": fb["fits"],
            "rf_vs_smote_holm20": next(r for r in results["per_classifier_holm20"]
                                      if r["classifier"] == "RF" and r["vs"] == "SMOTE")["holm_p"],
            "svml_vs_adasyn_holm20": next(r for r in results["per_classifier_holm20"]
                                         if r["classifier"] == "SVM-Linear" and r["vs"] == "ADASYN")["holm_p"],
            "rf_vs_smote_source_holm20": next(r for r in results["per_classifier_source_block_holm20"]
                                             if r["classifier"] == "RF" and r["vs"] == "SMOTE")["holm_p"],
            "svml_vs_adasyn_source_holm20": next(r for r in results["per_classifier_source_block_holm20"]
                                                if r["classifier"] == "SVM-Linear" and r["vs"] == "ADASYN")["holm_p"],
            "interaction_smote_source_holm4": next(r for r in results["interaction_rf_vs_svml"]["contrasts"] if r["vs"] == "SMOTE")["holm_p"],
            "interaction_smote_contrast": next(r for r in results["interaction_rf_vs_svml"]["contrasts"] if r["vs"] == "SMOTE")["mean_contrast"],
            "interaction_global_lrt_p": results["interaction_rf_vs_svml"]["global_lrt"]["p"],
            "clrobust_f1_baseline_p": next(r for r in results["cluster_robust_F1"] if r["vs"] == "Baseline")["raw_p"],
            "clrobust_f1_min_oversampler_p": min(r["raw_p"] for r in results["cluster_robust_F1"] if r["vs"] != "Baseline"),
            "clrobust_auprc_baseline_p": next(r for r in results["cluster_robust_AUPRC"] if r["vs"] == "Baseline")["raw_p"],
            "random_slope_auprc_baseline_p": results["random_slope_AUPRC"]["contrasts"]["Baseline"]["p"],
            "random_slope_auprc_adasyn_p": results["random_slope_AUPRC"]["contrasts"]["ADASYN"]["p"],
            "interaction_smote_holm24": next(r for r in results["interaction_all_pairs_holm24"] if r["pair"] == "RF vs SVM-Linear" and r["vs"] == "SMOTE")["holm_p"],
            "interaction_adasyn_holm24": next(r for r in results["interaction_all_pairs_holm24"] if r["pair"] == "RF vs SVM-Linear" and r["vs"] == "ADASYN")["holm_p"],
            "interaction_tomek_holm24": next(r for r in results["interaction_all_pairs_holm24"] if r["pair"] == "RF vs SVM-Linear" and r["vs"] == "SMOTE-Tomek")["holm_p"],
            "equiv_f1_smote_margin": next(r for r in results["equivalence_F1"] if r["vs"] == "SMOTE")["smallest_equivalence_margin"],
            "equiv_f1_adasyn_margin": next(r for r in results["equivalence_F1"] if r["vs"] == "ADASYN")["smallest_equivalence_margin"],
            "equiv_f1_max_noninferiority_margin": max(r["noninferiority_margin"] for r in results["equivalence_F1"]),
        }
        bad = 0
        print("\n" + "=" * 78)
        print("CHECK against manuscript values")
        print("=" * 78)
        for k, v in PAPER.items():
            g = got[k]
            ok = abs(g - v) <= max(5e-4, 0.03 * abs(v))
            bad += (not ok)
            print(f"  {'ok ' if ok else 'BAD'} {k:<36} paper={v:<12} computed={g:.6g}")
        print(f"\n{'all checks passed' if not bad else f'{bad} mismatch(es)'}")
        return 0 if not bad else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
