from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common.constants import STATUS_ELIGIBLE, STATUS_PENDING, SYNTHETIC_IMAGE_STRATEGIES

REGISTER_COLUMNS = [
    "Comparison ID",
    "Study ID",
    "Task",
    "Baseline Study Instance ID",
    "Baseline Source Row",
    "Baseline Method",
    "Metric",
    "Metric Family",
    "Baseline (%)",
    "Selected Study Instance ID",
    "Selected Source Result Row",
    "Proposed Method",
    "Mitigation (%)",
    "Status",
    "Reason",
    "Signed Gain (pp)",
]
NAMED_BASELINE_REASON = "The result names this baseline instance"
ONLY_BASELINE_REASON = "The study has one baseline"
NO_BASELINE_REASON = "No baseline for this study"
UNMATCHED_BASELINE_REASON = "The named baseline instance is not in the baseline sheet"
FAMILY_MISMATCH_REASON = "The baseline and the result use different metric families"
MISSING_SCORE_REASON = "Baseline or mitigation score is missing"
_COUNT = re.compile(r"\d{1,3}(?:,\d{3})+|\d+")


def _numeric_dataset_size(value):
    if isinstance(value, bool) or pd.isna(value):
        return pd.NA
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        if np.isfinite(value) and value >= 0 and value.is_integer():
            return int(value)
        return pd.NA
    text = str(value).strip().replace(" ", "")
    if _COUNT.fullmatch(text):
        return int(text.replace(",", ""))
    return pd.NA


def _strip_columns(frame, columns):
    for column in columns:
        frame[column] = frame[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    return frame


def _result_fields(row):
    if row is None:
        return {
            "Selected Study Instance ID": pd.NA,
            "Selected Source Result Row": pd.NA,
            "Proposed Method": pd.NA,
            "Mitigation (%)": pd.NA,
        }
    return {
        "Selected Study Instance ID": row["Study Instance ID"],
        "Selected Source Result Row": int(row["Source Result Row"]),
        "Proposed Method": row["Proposed Method"],
        "Mitigation (%)": row["Mitigation (%)"],
    }


def _named_baseline(value):
    if isinstance(value, str):
        return value.strip()
    if pd.isna(value):
        return ""
    return str(value).strip()


def _baseline_for_result(result, baselines, by_instance, baseline_counts):
    named = _named_baseline(result.get("Baseline Instance ID", pd.NA))
    if named:
        if named not in by_instance.index:
            return None, UNMATCHED_BASELINE_REASON
        chosen = by_instance.loc[named]
        if isinstance(chosen, pd.DataFrame):
            chosen = chosen.iloc[0]
        return chosen, NAMED_BASELINE_REASON
    if int(baseline_counts.get(result["Study ID"], 0)) == 1:
        return baselines.loc[baselines["Study ID"].eq(result["Study ID"])].iloc[0], ONLY_BASELINE_REASON
    return None, NO_BASELINE_REASON


def _comparison_register(baselines, results, master):
    tasks = master.set_index("Study ID")["Task"]
    by_instance = baselines.set_index("Baseline Study Instance ID", drop=False)
    baseline_counts = baselines.groupby("Study ID").size()
    rows = []
    for result in results.to_dict("records"):
        study_id = result["Study ID"]
        family = result["Metric Family"]
        chosen, reason = _baseline_for_result(result, baselines, by_instance, baseline_counts)
        status = STATUS_PENDING
        gain = pd.NA
        baseline_fields = {
            "Baseline Study Instance ID": pd.NA,
            "Baseline Source Row": pd.NA,
            "Baseline Method": pd.NA,
            "Baseline (%)": pd.NA,
        }
        if chosen is not None:
            baseline_fields = {
                "Baseline Study Instance ID": chosen["Baseline Study Instance ID"],
                "Baseline Source Row": int(chosen["Baseline Source Row"]),
                "Baseline Method": chosen["Baseline Method"],
                "Baseline (%)": chosen["Baseline (%)"],
            }
            same_family = str(chosen["Metric Family"]).strip() == str(family).strip()
            scores_present = pd.notna(chosen["Baseline (%)"]) and pd.notna(result["Mitigation (%)"])
            if same_family and scores_present:
                status = STATUS_ELIGIBLE
                gain = float(result["Mitigation (%)"]) - float(chosen["Baseline (%)"])
            elif not same_family:
                reason = FAMILY_MISMATCH_REASON
            else:
                reason = MISSING_SCORE_REASON
        rows.append({
            "Study ID": study_id,
            "Task": tasks.get(study_id, pd.NA),
            "Metric": family,
            "Metric Family": family,
            "Status": status,
            "Reason": reason,
            "Signed Gain (pp)": gain,
            **baseline_fields,
            **_result_fields(result),
        })
    register = pd.DataFrame(rows)
    register.insert(0, "Comparison ID", [f"C{i:04d}" for i in range(1, len(register) + 1)])
    register["Signed Gain (pp)"] = pd.to_numeric(register["Signed Gain (pp)"], errors="coerce")
    register["Selected Source Result Row"] = pd.to_numeric(register["Selected Source Result Row"], errors="coerce").astype("Int64")
    return register[REGISTER_COLUMNS]


def _audit(results, register):
    chosen = register.loc[
        register["Status"].eq(STATUS_ELIGIBLE) & register["Selected Source Result Row"].notna(),
        ["Selected Source Result Row", "Comparison ID", "Baseline Study Instance ID"],
    ].copy()
    chosen = chosen.groupby("Selected Source Result Row", as_index=False).agg({
        "Comparison ID": lambda values: "; ".join(values.astype(str)),
        "Baseline Study Instance ID": lambda values: "; ".join(values.astype(str)),
    })
    audit = results.merge(chosen, left_on="Source Result Row", right_on="Selected Source Result Row", how="left")
    audit["Selected"] = np.where(audit["Comparison ID"].notna(), "Yes", "No")
    audit = audit.drop(columns=["Selected Source Result Row"])
    return audit


def build_analysis_ready(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    ident = raw["identification"].copy()
    context = raw["context"].copy().rename(columns={"Crop ": "Crop"})
    datasets = raw["datasets"].copy()
    master = ident.merge(context, on="Study ID", how="left", validate="one_to_one")
    master = master.merge(datasets, on="Study ID", how="left", validate="one_to_one")
    master["Dataset Size (numeric)"] = master["Dataset Size"].map(_numeric_dataset_size).astype("Int64")
    master["Year"] = pd.to_numeric(master["Year"], errors="coerce").astype("Int64")

    barriers = _strip_columns(raw["barriers"].copy().rename(columns={
        "Modules": "Barrier Modules",
        "Barriers": "Normalized Barrier",
    }), ["Study ID", "Study Instance ID"])
    mitigations = _strip_columns(raw["mitigations"].copy().rename(columns={
        "Mitigations": "Mitigation Strategy",
    }), ["Study ID", "Study Instance ID"])
    task = master[["Study ID", "Task"]]
    barrier_profile = barriers.merge(task, on="Study ID", how="left", validate="many_to_one")
    mitigations["Synthetic Image Method"] = mitigations["Mitigation Strategy"].isin(SYNTHETIC_IMAGE_STRATEGIES).map({
        True: "Yes",
        False: "No",
    })
    mitigations["Synthetic Type"] = mitigations["Mitigation Strategy"].where(mitigations["Synthetic Image Method"].eq("Yes"), "")
    mitigation_profile = mitigations.merge(task, on="Study ID", how="left", validate="many_to_one")

    key = ["Study ID", "Study Instance ID"]
    cooccurrence = barriers.merge(mitigations, on=key, how="outer", validate="one_to_many", indicator=True)
    cooccurrence["Co-occurrence"] = cooccurrence["_merge"].map({
        "both": "Same-instance co-occurrence",
        "left_only": "Barrier only",
        "right_only": "Mitigation only",
    })
    cooccurrence = cooccurrence.drop(columns="_merge")
    cooccurrence = cooccurrence.merge(task, on="Study ID", how="left", validate="many_to_one")

    baselines = _strip_columns(raw["baselines"].copy().rename(columns={
        "Baseline Modules": "Baseline Method",
        "Study Instance ID": "Baseline Study Instance ID",
    }), ["Study ID", "Baseline Study Instance ID", "Metric Family"])
    baselines["Baseline (%)"] = pd.to_numeric(baselines["Baseline (%)"], errors="coerce")
    baselines = baselines.reset_index(drop=True)
    baselines["Baseline Source Row"] = np.arange(1, len(baselines) + 1)

    results = raw["mitigation_results"].copy().rename(columns={
        "Mitigation Modules": "Proposed Method",
    })
    result_text = ["Study ID", "Study Instance ID", "Metric Family"]
    if "Baseline Instance ID" in results.columns:
        result_text.append("Baseline Instance ID")
    results = _strip_columns(results, result_text)
    if "Gain (pp)" in results.columns:
        results = results.rename(columns={"Gain (pp)": "Reported Gain (pp)"})
    elif "Gain" in results.columns:
        results = results.rename(columns={"Gain": "Reported Gain (pp)"})
    else:
        results["Reported Gain (pp)"] = pd.NA
    results["Mitigation (%)"] = pd.to_numeric(results["Mitigation (%)"], errors="coerce")
    results["Reported Gain (pp)"] = pd.to_numeric(results["Reported Gain (pp)"], errors="coerce")
    results = results.reset_index(drop=True)
    results["Source Result Row"] = np.arange(1, len(results) + 1)

    register = _comparison_register(baselines, results, master)
    audit = _audit(results, register)
    performance = register.loc[register["Status"].eq(STATUS_ELIGIBLE)].reset_index(drop=True)

    quality_raw = raw["quality"].copy()
    quality_columns = {
        "Is the specific training-data barrier(s) clearly identified and described?": "Barrier Clearly Identified",
        "Is the prevalence or severity of the barrier quantified or well justified?": "Barrier Severity Justified",
        "Is the mitigation strategy clearly named and described?": "Mitigation Clearly Described",
        "Is the link between the barrier and the chosen strategy explicitly justified?": "Barrier-Mitigation Link Justified",
        "Are absolute performance values (before and after mitigation) reported with clear metrics?": "Absolute Performance Reported",
        "Is the baseline performance (without the mitigation) clearly reported?": "Baseline Clearly Reported",
        "Is the performance gain calculated and reported transparently?": "Performance Gain Transparent",
        "Are performance results supported by uncertainty/variability evidence such as repeated runs, folds, SD, CI, or multiple seeds?": "Uncertainty Reported",
        "Is the experimental environment clearly stated (field / UAV / greenhouse / lab / controlled)?": "Environment Clearly Stated",
        "Is dataset size (number of images or instances) clearly reported?": "Dataset Size Clearly Reported",
        "Is the computer-vision task clearly specified (Detection / Classification / Segmentation)?": "Task Clearly Specified",
        "Is the evaluation dataset reasonably representative of the intended operating conditions?": "Representative Evaluation",
        "Was a proper independent test set (or external validation) used?": "Independent Test Set",
        "Was data leakage prevented?": "Data Leakage Prevented",
        "Were appropriate metrics used for the task?": "Appropriate Metrics",
        "Are limitations of the training data and mitigation strategy discussed?": "Limitations Discussed",
        "Are training/validation/test samples separated sufficiently to avoid closely related images or acquisition groups appearing across splits?": "Related Samples Separated",
        "If synthetic images were used, is the generation method clearly described?": "Synthetic Generation Described",
        "Is a fair comparison with non-synthetic alternatives provided?": "Fair Non-Synthetic Comparison",
        "Domain-level Concern": "Domain-level Concern",
        "Overall Risk of Bias": "Overall Risk of Bias",
        "Applicability / Generalizability Concern": "Applicability / Generalizability Concern",
    }
    quality = quality_raw.rename(columns=quality_columns)
    quality = quality.merge(master[["Study ID", "Title", "Task"]], on="Study ID", how="left", validate="one_to_one")
    for column in quality_columns.values():
        quality[column] = quality[column].fillna("Not reported").astype(str).str.strip()

    return {
        "study_master": master,
        "barrier_profile": barrier_profile,
        "mitigation_profile": mitigation_profile,
        "barrier_mitigation_cooccurrence": cooccurrence,
        "comparison_register": register,
        "performance_results": performance,
        "performance_linkage_audit": audit,
        "evidence_quality": quality,
    }
