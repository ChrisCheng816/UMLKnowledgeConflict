#!/usr/bin/env python3
"""Library-first evaluation pipeline for experiment_results.

Included methods (independent):
- passk_count_strength
- model_size_vs_min_mcnemar

Notes:
- Automatically repairs outputs_task2 via sanitize_outputs_task2.py.
- Never modifies original experiment_results files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Tuple

REQUIRED_MODULES = ["numpy", "pandas", "statsmodels"]


def ensure_dependencies() -> None:
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        msg = (
            "Missing required Python packages: "
            + ", ".join(missing)
            + "\nInstall with:\n"
            + "python3 -m pip install numpy pandas statsmodels"
        )
        raise SystemExit(msg)


ensure_dependencies()

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from sanitize_outputs_task2 import sanitize_outputs_task2

FILE_RE = re.compile(r"(.+?)_(forward|reverse|mixed)_([a-z]+)_(\d+)\.jsonl$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation analyses (passk_count_strength + model_size_vs_min_mcnemar)."
    )
    parser.add_argument("--input-root", default="/scratch/zcheng06/Test/experiment_results")
    parser.add_argument("--output-dir", default="/scratch/zcheng06/Test/eval")
    parser.add_argument(
        "--methods",
        default="passk_count_strength,model_size_vs_min_mcnemar",
        help="Comma-separated: passk_count_strength,model_size_vs_min_mcnemar",
    )
    parser.add_argument("--passk-min-draws", type=int, default=10)
    parser.add_argument(
        "--model-compare-metrics",
        default="pass10",
        help="Comma-separated binary metrics for model_size_vs_min_mcnemar: pass1,pass5,pass10",
    )
    return parser.parse_args()


def parse_bool(v: object) -> int:
    return int(str(v).strip().lower() == "true")


def classify_raw_task2_value(v: object) -> str:
    """Classify raw outputs_task2 value before sanitize step."""
    if isinstance(v, bool):
        return "True" if v else "False"
    if v is None:
        return "Dirty"
    s = str(v).strip()
    if s in {"True", "False", "Unknown"}:
        return s
    return "Dirty"


def save_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def save_csv_rounded(df: pd.DataFrame, path: str, digits: int = 4) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = df.copy()
    float_cols = out.select_dtypes(include=["float32", "float64"]).columns
    if len(float_cols) > 0:
        out[float_cols] = out[float_cols].round(digits)
    out.to_csv(path, index=False, float_format=f"%.{digits}f")


def iter_result_files(input_root: str):
    for model in sorted(os.listdir(input_root)):
        model_dir = os.path.join(input_root, model)
        if not os.path.isdir(model_dir):
            continue
        for fn in sorted(os.listdir(model_dir)):
            if not fn.endswith(".jsonl"):
                continue
            m = FILE_RE.match(fn)
            if not m:
                continue
            _, direction, relation, class_count_text = m.groups()
            class_count = int(class_count_text)
            yield model, direction, relation, class_count, os.path.join(model_dir, fn)


def load_dataframe(
    input_root: str,
    exclude_forward_3: bool,
    repair_passk_monotonic: bool,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    monotonic_violations = Counter()
    sanitize_rule_counts = Counter()

    total_files = 0

    for model, direction, relation, class_count, fp in iter_result_files(input_root):
        if exclude_forward_3 and direction == "forward" and class_count == 3:
            continue
        total_files += 1

        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                pass1 = parse_bool(obj.get("pass@1"))
                pass5 = parse_bool(obj.get("pass@5"))
                pass10 = parse_bool(obj.get("pass@10"))

                if pass1 and not pass5:
                    monotonic_violations["pass@1>pass@5"] += 1
                if pass5 and not pass10:
                    monotonic_violations["pass@5>pass@10"] += 1

                if repair_passk_monotonic:
                    pass5 = max(pass5, pass1)
                    pass10 = max(pass10, pass5)

                outputs = obj.get("outputs_task2", [])
                if not isinstance(outputs, list):
                    outputs = [outputs]

                raw_counts = Counter(classify_raw_task2_value(v) for v in outputs)
                repaired_labels, fallback_indices, repair_rules = sanitize_outputs_task2(outputs)
                sanitize_rule_counts.update(repair_rules)
                dirty_to_true = 0
                dirty_to_false = 0
                dirty_to_unknown = 0
                for raw_v, repaired_v in zip(outputs, repaired_labels):
                    if classify_raw_task2_value(raw_v) != "Dirty":
                        continue
                    if repaired_v == "True":
                        dirty_to_true += 1
                    elif repaired_v == "False":
                        dirty_to_false += 1
                    else:
                        dirty_to_unknown += 1

                outputs_task1 = obj.get("outputs_task1", [])
                if not isinstance(outputs_task1, list):
                    outputs_task1 = [outputs_task1]
                task1_any_true = int(any(parse_bool(v) == 1 for v in outputs_task1))

                draw_counts = Counter(repaired_labels)
                draw_true_k5 = sum(1 for x in repaired_labels[:5] if x == "True")
                draw_true_k10 = sum(1 for x in repaired_labels[:10] if x == "True")
                sample_id = str(obj.get("id"))

                rows.append(
                    {
                        "model": model,
                        "direction": direction,
                        "relation": relation,
                        "class_count": class_count,
                        "sample_id": sample_id,
                        "pass1": pass1,
                        "pass5": pass5,
                        "pass10": pass10,
                        "task1_pass1": parse_bool(obj.get("task1_pass@1")),
                        "task1_pass5": parse_bool(obj.get("task1_pass@5")),
                        "task1_pass10": parse_bool(obj.get("task1_pass@10")),
                        "task1_any_true": task1_any_true,
                        "draw_len": len(repaired_labels),
                        "draw_true_before": raw_counts.get("True", 0),
                        "draw_false_before": raw_counts.get("False", 0),
                        "draw_unknown_before": raw_counts.get("Unknown", 0),
                        "draw_dirty_before": raw_counts.get("Dirty", 0),
                        "draw_true": draw_counts.get("True", 0),
                        "draw_false": draw_counts.get("False", 0),
                        "draw_unknown": draw_counts.get("Unknown", 0),
                        "draw_true_k5": draw_true_k5,
                        "draw_true_k10": draw_true_k10,
                        "draw_fallback_count": len(fallback_indices),
                        "dirty_to_true": dirty_to_true,
                        "dirty_to_false": dirty_to_false,
                        "dirty_to_unknown": dirty_to_unknown,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No rows loaded from input-root after filters.")

    meta = {
        "total_files": total_files,
        "total_rows": int(len(df)),
        "exclude_forward_3": bool(exclude_forward_3),
        "repair_passk_monotonic": bool(repair_passk_monotonic),
        "monotonic_violations_before_repair": dict(monotonic_violations),
        "sanitize_rule_counts": dict(sanitize_rule_counts),
    }
    return df, meta


def _mcnemar_or(a: pd.Series, b: pd.Series) -> Dict[str, float]:
    table = pd.crosstab(a, b).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    n00 = int(table.loc[0, 0])
    n01 = int(table.loc[0, 1])
    n10 = int(table.loc[1, 0])
    n11 = int(table.loc[1, 1])

    res = mcnemar(table, exact=True, correction=False)
    p_exact = float(res.pvalue)

    odds_ratio = (n10 + 0.5) / (n01 + 0.5)
    log_or = np.log(odds_ratio)
    se = np.sqrt(1.0 / (n10 + 0.5) + 1.0 / (n01 + 0.5))
    or_ci_low = float(np.exp(log_or - 1.96 * se))
    or_ci_high = float(np.exp(log_or + 1.96 * se))

    return {
        "n": int(len(a)),
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "discordant": int(n01 + n10),
        "mcnemar_p_exact": p_exact,
        "odds_ratio_n10_over_n01": float(odds_ratio),
        "odds_ratio_ci_low": or_ci_low,
        "odds_ratio_ci_high": or_ci_high,
    }


def _parse_model_series_and_size(model: str) -> Tuple[str, float] | None:
    m = re.match(r"^(.*)-(\d+(?:\.\d+)?)[bB]$", model.strip())
    if not m:
        return None
    series = m.group(1).strip()
    size_b = float(m.group(2))
    return series, size_b


def run_model_size_vs_min_mcnemar(df: pd.DataFrame, outdir: str, min_draws: int, metrics: List[str]) -> None:
    draw_min = df.groupby("model")["draw_len"].min()
    eligible_models = set(draw_min[draw_min >= min_draws].index)
    sdf = df[df["model"].isin(eligible_models)].copy()

    parsed = sdf["model"].map(_parse_model_series_and_size)
    sdf["model_series"] = parsed.map(lambda x: x[0] if x is not None else np.nan)
    sdf["model_size_b"] = parsed.map(lambda x: x[1] if x is not None else np.nan)
    sdf = sdf.dropna(subset=["model_series", "model_size_b"]).copy()

    group_keys = ["model_series", "direction", "relation", "class_count"]
    rows = []

    for key, sub in sdf.groupby(group_keys, sort=True):
        if not isinstance(key, tuple):
            key = (key,)

        model_sizes = (
            sub[["model", "model_size_b"]]
            .drop_duplicates()
            .sort_values(["model_size_b", "model"], ascending=[True, True])
        )
        if len(model_sizes) < 2:
            continue
        base_model = str(model_sizes.iloc[0]["model"])
        base_size = float(model_sizes.iloc[0]["model_size_b"])
        larger_models = [str(m) for m in model_sizes["model"].tolist()[1:]]

        for metric in metrics:
            pivot = sub.pivot_table(index="sample_id", columns="model", values=metric, aggfunc="first")
            if base_model not in pivot.columns:
                continue

            group_metric_rows = []
            for cmp_model in larger_models:
                if cmp_model not in pivot.columns:
                    continue
                pair = pivot[[base_model, cmp_model]].dropna()
                if pair.empty:
                    continue

                x = pair[base_model].astype(int)
                y = pair[cmp_model].astype(int)
                cmp_size = float(model_sizes[model_sizes["model"] == cmp_model]["model_size_b"].iloc[0])

                rec = {group_keys[idx]: key[idx] for idx in range(len(group_keys))}
                rec["metric"] = metric
                rec["base_model"] = base_model
                rec["base_size_b"] = base_size
                rec["compare_model"] = cmp_model
                rec["compare_size_b"] = cmp_size
                rec["n_paired"] = int(len(pair))
                rec["rate_base"] = float(x.mean())
                rec["rate_compare"] = float(y.mean())
                rec["delta_rate_compare_minus_base"] = float(y.mean() - x.mean())
                rec.update(_mcnemar_or(x, y))
                rec["odds_ratio_compare_over_base"] = (
                    float(1.0 / rec["odds_ratio_n10_over_n01"]) if rec["odds_ratio_n10_over_n01"] != 0 else np.nan
                )
                group_metric_rows.append(rec)

            if group_metric_rows:
                pvals = [r["mcnemar_p_exact"] for r in group_metric_rows]
                p_holm = multipletests(pvals, method="holm")[1]
                for rec, p_adj in zip(group_metric_rows, p_holm):
                    rec["mcnemar_p_holm"] = float(p_adj)
                    rows.append(rec)

    out = pd.DataFrame(rows)
    drop_cols = [
        "n_paired",
        "rate_base",
        "rate_compare",
        "delta_rate_compare_minus_base",
        "n",
        "n00",
        "n01",
        "n10",
        "n11",
        "discordant",
        "odds_ratio_n10_over_n01",
        "odds_ratio_ci_low",
        "odds_ratio_ci_high",
    ]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])
    if "odds_ratio_compare_over_base" in out.columns:
        out = out.rename(columns={"odds_ratio_compare_over_base": "odds_ratio"})
    if "base_size_b" in out.columns:
        out["base_size_b"] = out["base_size_b"].astype(int)
    if "compare_size_b" in out.columns:
        out["compare_size_b"] = out["compare_size_b"].astype(int)
    save_csv_rounded(
        out,
        os.path.join(outdir, "size_mcnemar.csv"),
        digits=4,
    )


def _count_strength_summary(sub: pd.DataFrame) -> Dict[str, float]:
    n = int(len(sub))
    out: Dict[str, float] = {"n": n}
    if n == 0:
        out.update(
            {
                "mean_true_count": np.nan,
                "median_true_count": np.nan,
                "std_true_count": np.nan,
                "mean_true_rate": np.nan,
                "n_pass10_true": 0,
                "mean_true_count_given_pass10_true": np.nan,
                "mean_true_rate_given_pass10_true": np.nan,
            }
        )
        return out

    tc = sub["draw_true"].astype(float)
    dr = sub["draw_len"].astype(float)
    mask_p10 = sub["pass10"].astype(int) == 1

    out.update(
        {
            "mean_true_count": float(tc.mean()),
            "median_true_count": float(tc.median()),
            "std_true_count": float(tc.std(ddof=1)) if n > 1 else np.nan,
            "mean_true_rate": float((tc / dr).mean()),
            "n_pass10_true": int(mask_p10.sum()),
        }
    )

    if mask_p10.any():
        tc2 = tc[mask_p10]
        dr2 = dr[mask_p10]
        out["mean_true_count_given_pass10_true"] = float(tc2.mean())
        out["mean_true_rate_given_pass10_true"] = float((tc2 / dr2).mean())
    else:
        out["mean_true_count_given_pass10_true"] = np.nan
        out["mean_true_rate_given_pass10_true"] = np.nan
    return out


def run_passk_count_strength(df: pd.DataFrame, outdir: str, min_draws: int) -> None:
    draw_min = df.groupby("model")["draw_len"].min()
    eligible_models = set(draw_min[draw_min >= min_draws].index)
    sdf = df[df["model"].isin(eligible_models)].copy()

    # Summary: by model + condition, reporting True rates for pass@k and task1_pass@k.
    keys = ["model", "direction", "relation", "class_count"]
    rows_cond = []
    for key, sub in sdf.groupby(keys, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        rec = {keys[i]: key[i] for i in range(len(keys))}
        rec["n"] = int(len(sub))
        rec["pass1"] = float(sub["pass1"].astype(int).mean())
        rec["pass5"] = float(sub["pass5"].astype(int).mean())
        rec["pass10"] = float(sub["pass10"].astype(int).mean())
        rec["task1_pass1"] = float(sub["task1_pass1"].astype(int).mean())
        rec["task1_pass5"] = float(sub["task1_pass5"].astype(int).mean())
        rec["task1_pass10"] = float(sub["task1_pass10"].astype(int).mean())
        rec["mean_true_count_k5"] = float(sub["draw_true_k5"].astype(float).mean())
        rec["mean_true_count_k10"] = float(sub["draw_true_k10"].astype(float).mean())
        # Conditional pass@k rates over rows whose outputs_task1 contains at least one True.
        mask_task1_true = sub["task1_any_true"].astype(int) == 1
        rec["n_t1true"] = int(mask_task1_true.sum())
        if bool(mask_task1_true.any()):
            rec["p1_t1true"] = float(
                sub.loc[mask_task1_true, "pass1"].astype(int).mean()
            )
            rec["p5_t1true"] = float(
                sub.loc[mask_task1_true, "pass5"].astype(int).mean()
            )
            rec["p10_t1true"] = float(
                sub.loc[mask_task1_true, "pass10"].astype(int).mean()
            )
        else:
            rec["p1_t1true"] = np.nan
            rec["p5_t1true"] = np.nan
            rec["p10_t1true"] = np.nan

        rec["p5_p1_uplift_pct"] = float((rec["pass5"] - rec["pass1"]) * 100.0)
        rec["p10_p1_uplift_pct"] = float((rec["pass10"] - rec["pass1"]) * 100.0)
        if pd.notna(rec["p1_t1true"]):
            rec["p5_p1_t1true_uplift_pct"] = float(
                (rec["p5_t1true"] - rec["p1_t1true"]) * 100.0
            )
            rec["p10_p1_t1true_uplift_pct"] = float(
                (rec["p10_t1true"] - rec["p1_t1true"]) * 100.0
            )
        else:
            rec["p5_p1_t1true_uplift_pct"] = np.nan
            rec["p10_p1_t1true_uplift_pct"] = np.nan
        rows_cond.append(rec)
    by_model_cond = pd.DataFrame(rows_cond)
    prob_cols = [
        "pass1",
        "pass5",
        "pass10",
        "task1_pass1",
        "task1_pass5",
        "task1_pass10",
        "p1_t1true",
        "p5_t1true",
        "p10_t1true",
    ]
    mean_count_cols = ["mean_true_count_k5", "mean_true_count_k10"]
    uplift_cols = [
        "p5_p1_uplift_pct",
        "p10_p1_uplift_pct",
        "p5_p1_t1true_uplift_pct",
        "p10_p1_t1true_uplift_pct",
    ]
    for col in prob_cols:
        if col in by_model_cond.columns:
            by_model_cond[col] = by_model_cond[col].map(
                lambda v: f"{float(v):.4f}" if pd.notna(v) else ""
            )
    for col in mean_count_cols:
        if col in by_model_cond.columns:
            by_model_cond[col] = by_model_cond[col].map(
                lambda v: f"{float(v):.4f}" if pd.notna(v) else ""
            )
    for col in uplift_cols:
        if col in by_model_cond.columns:
            by_model_cond[col] = by_model_cond[col].map(
                lambda v: f"{float(v):.2f}" if pd.notna(v) else ""
            )
    os.makedirs(outdir, exist_ok=True)
    by_model_cond.to_csv(os.path.join(outdir, "passk_count.csv"), index=False)

def build_preprocess_detailed_stats(df: pd.DataFrame, meta: Dict[str, object]) -> Dict[str, object]:
    overall = {
        "n_rows": int(len(df)),
        "n_models": int(df["model"].nunique()),
        "n_conditions": int(df[["direction", "relation", "class_count"]].drop_duplicates().shape[0]),
        "pass1_rate": float(df["pass1"].mean()),
        "pass5_rate": float(df["pass5"].mean()),
        "pass10_rate": float(df["pass10"].mean()),
        "task1_pass1_rate": float(df["task1_pass1"].mean()),
        "draw_true_before_mean": float(df["draw_true_before"].mean()),
        "draw_false_before_mean": float(df["draw_false_before"].mean()),
        "draw_unknown_before_mean": float(df["draw_unknown_before"].mean()),
        "draw_dirty_before_mean": float(df["draw_dirty_before"].mean()),
        "draw_true_mean": float(df["draw_true"].mean()),
        "draw_false_mean": float(df["draw_false"].mean()),
        "draw_unknown_mean": float(df["draw_unknown"].mean()),
        "draw_dirty_before_sum": int(df["draw_dirty_before"].sum()),
        "draw_fallback_count_sum": int(df["draw_fallback_count"].sum()),
        "dirty_to_true_sum": int(df["dirty_to_true"].sum()),
        "dirty_to_false_sum": int(df["dirty_to_false"].sum()),
        "dirty_to_unknown_sum": int(df["dirty_to_unknown"].sum()),
    }
    if overall["draw_dirty_before_sum"] > 0:
        overall["dirty_to_true_ratio"] = float(
            overall["dirty_to_true_sum"] / overall["draw_dirty_before_sum"]
        )
        overall["dirty_to_false_ratio"] = float(
            overall["dirty_to_false_sum"] / overall["draw_dirty_before_sum"]
        )
        overall["dirty_to_unknown_ratio"] = float(
            overall["dirty_to_unknown_sum"] / overall["draw_dirty_before_sum"]
        )
    else:
        overall["dirty_to_true_ratio"] = np.nan
        overall["dirty_to_false_ratio"] = np.nan
        overall["dirty_to_unknown_ratio"] = np.nan

    by_model = (
        df.groupby("model", sort=True)
        .agg(
            n=("sample_id", "size"),
            pass1_rate=("pass1", "mean"),
            pass5_rate=("pass5", "mean"),
            pass10_rate=("pass10", "mean"),
            task1_pass1_rate=("task1_pass1", "mean"),
            draw_true_before_mean=("draw_true_before", "mean"),
            draw_false_before_mean=("draw_false_before", "mean"),
            draw_unknown_before_mean=("draw_unknown_before", "mean"),
            draw_dirty_before_mean=("draw_dirty_before", "mean"),
            draw_true_mean=("draw_true", "mean"),
            draw_false_mean=("draw_false", "mean"),
            draw_unknown_mean=("draw_unknown", "mean"),
            draw_dirty_before_sum=("draw_dirty_before", "sum"),
            draw_fallback_count_sum=("draw_fallback_count", "sum"),
            dirty_to_true_sum=("dirty_to_true", "sum"),
            dirty_to_false_sum=("dirty_to_false", "sum"),
            dirty_to_unknown_sum=("dirty_to_unknown", "sum"),
        )
        .reset_index()
    )
    by_model["dirty_to_true_ratio"] = np.where(
        by_model["draw_dirty_before_sum"] > 0,
        by_model["dirty_to_true_sum"] / by_model["draw_dirty_before_sum"],
        np.nan,
    )
    by_model["dirty_to_false_ratio"] = np.where(
        by_model["draw_dirty_before_sum"] > 0,
        by_model["dirty_to_false_sum"] / by_model["draw_dirty_before_sum"],
        np.nan,
    )
    by_model["dirty_to_unknown_ratio"] = np.where(
        by_model["draw_dirty_before_sum"] > 0,
        by_model["dirty_to_unknown_sum"] / by_model["draw_dirty_before_sum"],
        np.nan,
    )

    by_model_condition = (
        df.groupby(["model", "direction", "relation", "class_count"], sort=True)
        .agg(
            n=("sample_id", "size"),
            pass1_rate=("pass1", "mean"),
            pass5_rate=("pass5", "mean"),
            pass10_rate=("pass10", "mean"),
            task1_pass1_rate=("task1_pass1", "mean"),
            draw_true_before_mean=("draw_true_before", "mean"),
            draw_false_before_mean=("draw_false_before", "mean"),
            draw_unknown_before_mean=("draw_unknown_before", "mean"),
            draw_dirty_before_mean=("draw_dirty_before", "mean"),
            draw_true_mean=("draw_true", "mean"),
            draw_false_mean=("draw_false", "mean"),
            draw_unknown_mean=("draw_unknown", "mean"),
            draw_dirty_before_sum=("draw_dirty_before", "sum"),
            draw_fallback_count_sum=("draw_fallback_count", "sum"),
            dirty_to_true_sum=("dirty_to_true", "sum"),
            dirty_to_false_sum=("dirty_to_false", "sum"),
            dirty_to_unknown_sum=("dirty_to_unknown", "sum"),
        )
        .reset_index()
    )
    by_model_condition["dirty_to_true_ratio"] = np.where(
        by_model_condition["draw_dirty_before_sum"] > 0,
        by_model_condition["dirty_to_true_sum"] / by_model_condition["draw_dirty_before_sum"],
        np.nan,
    )
    by_model_condition["dirty_to_false_ratio"] = np.where(
        by_model_condition["draw_dirty_before_sum"] > 0,
        by_model_condition["dirty_to_false_sum"] / by_model_condition["draw_dirty_before_sum"],
        np.nan,
    )
    by_model_condition["dirty_to_unknown_ratio"] = np.where(
        by_model_condition["draw_dirty_before_sum"] > 0,
        by_model_condition["dirty_to_unknown_sum"] / by_model_condition["draw_dirty_before_sum"],
        np.nan,
    )
    by_model = by_model.round(2)
    by_model_condition = by_model_condition.round(2)
    for k, v in list(overall.items()):
        if isinstance(v, float):
            overall[k] = round(v, 2)

    fallback_handling = {
        "sanitizer": "sanitize_outputs_task2.py",
        "rule_counts": meta.get("sanitize_rule_counts", {}),
        "description": "Rule counts are global; draw_fallback_count_sum gives fallback volume. dirty_to_*_ratio gives how dirty values were repaired by model/config.",
    }

    return {
        "overall": overall,
        "data_meta": meta,
        "fallback_handling": fallback_handling,
        "by_model": json.loads(by_model.to_json(orient="records", force_ascii=True)),
        "by_model_condition": json.loads(by_model_condition.to_json(orient="records", force_ascii=True)),
    }


def main() -> int:
    args = parse_args()
    exclude_forward_3 = True
    repair_passk_monotonic = True
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    valid = {"passk_count_strength", "model_size_vs_min_mcnemar"}
    model_metrics = [m.strip() for m in args.model_compare_metrics.split(",") if m.strip()]
    valid_metrics = {"pass1", "pass5", "pass10"}

    unknown = [m for m in methods if m not in valid]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")
    unknown_metrics = [m for m in model_metrics if m not in valid_metrics]
    if unknown_metrics:
        raise SystemExit(f"Unknown model-compare-metrics: {unknown_metrics}")

    os.makedirs(args.output_dir, exist_ok=True)

    df, meta = load_dataframe(
        input_root=args.input_root,
        exclude_forward_3=exclude_forward_3,
        repair_passk_monotonic=repair_passk_monotonic,
    )
    save_json(
        os.path.join(args.output_dir, "prepstats.json"),
        build_preprocess_detailed_stats(df, meta),
    )

    run_meta = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": args.input_root,
        "output_dir": args.output_dir,
        "methods": methods,
        "args": vars(args),
        "fixed_behavior": {
            "exclude_forward_3": exclude_forward_3,
            "repair_passk_monotonic": repair_passk_monotonic,
        },
        "preprocess_outputs": {
            "detailed_stats_json": "prepstats.json",
        },
        "data_meta": meta,
        "num_rows_loaded": int(len(df)),
    }
    save_json(os.path.join(args.output_dir, "runmeta.json"), run_meta)

    for method in methods:
        if method == "passk_count_strength":
            run_passk_count_strength(df, args.output_dir, min_draws=args.passk_min_draws)
        elif method == "model_size_vs_min_mcnemar":
            run_model_size_vs_min_mcnemar(
                df,
                args.output_dir,
                min_draws=args.passk_min_draws,
                metrics=model_metrics,
            )

    print(f"Done. Wrote outputs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
