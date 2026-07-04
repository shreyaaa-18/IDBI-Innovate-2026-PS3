"""
MSME Financial Health Score — Explainability Engine
-------------------------------------------------------

This scoring engine is a transparent, weighted rules-based model (not a
black-box ML model), we don't need to approximate feature importance the way
SHAP does for opaque models — we can compute EXACT, per-feature point
contributions directly from the scoring formula itself. This is arguably a
stronger explainability story than SHAP for a banking use case: every point
in the score can be traced to a specific number, with no approximation error.

Method — "deviation-from-average" attribution:
  For each feature, we compare THIS business's normalized value to the
  population AVERAGE normalized value, and multiply the difference by that
  feature's effective weight (which already accounts for the adaptive
  data-maturity adjustment from score_engine_adaptive.py).

  contribution_points(feature) = (business_value - population_avg_value)
                                   x effective_weight x 6

  This tells a clean story: "Compared to a typical MSME, this business's
  GST filing behavior added +18 points to its score" — which sums up
  EXACTLY to the business's actual score. No black box, no approximation.

Output: msme_explanations.json — one entry per business with top positive/
negative drivers and a plain-English narrative, ready for the dashboard.
"""

import json
import numpy as np
import pandas as pd

IN_PATH = "msme_synthetic_dataset.csv"
OUT_SCORED_PATH = "msme_scored_dataset_final.csv"
OUT_EXPLANATIONS_PATH = "msme_explanations.json"

# ---------------------------------------------------------------------------
# Same feature config as score_engine_adaptive.py
# (col, direction, base_weight_within_pillar, vintage_dependent)
# ---------------------------------------------------------------------------
PILLAR_FEATURES = {
    "cash_flow": [
        ("monthly_upi_volume_inr", "higher", 0.30, False),
        ("avg_monthly_balance_inr", "higher", 0.25, False),
        ("txn_count_per_month", "higher", 0.15, False),
        ("bounce_rate_pct", "lower", 0.15, False),
        ("balance_volatility_pct", "lower", 0.15, False),
    ],
    "compliance": [
        ("gst_on_time_filing_pct", "higher", 0.40, False),
        ("gst_vs_bank_turnover_gap_pct", "lower", 0.30, False),
        ("months_since_last_gst_filing", "lower", 0.15, False),
        ("gst_registration_vintage_years", "higher", 0.15, True),
    ],
    "stability": [
        ("employee_count", "higher", 0.20, False),
        ("employee_retention_rate_pct", "higher", 0.35, False),
        ("pf_contribution_regularity_pct", "higher", 0.30, False),
        ("avg_salary_payment_delay_days", "lower", 0.15, False),
    ],
    "growth": [
        ("revenue_6m_growth_pct", "higher", 0.65, False),
        ("revenue_volatility_pct", "lower", 0.35, False),
    ],
}

BASE_PILLAR_WEIGHTS = {"cash_flow": 0.30, "compliance": 0.25, "stability": 0.20, "growth": 0.25}
PILLAR_REDISTRIBUTION_SPLIT = {"cash_flow": 0.6, "growth": 0.4}
MATURITY_LOW_YEARS = 0.25
MATURITY_HIGH_YEARS = 2.5

# Human-readable labels + short explanation templates for the dashboard/demo
FEATURE_LABELS = {
    "monthly_upi_volume_inr": "Monthly UPI transaction volume",
    "avg_monthly_balance_inr": "Average monthly bank balance",
    "txn_count_per_month": "Transaction frequency",
    "bounce_rate_pct": "Payment bounce rate",
    "balance_volatility_pct": "Balance stability",
    "gst_on_time_filing_pct": "On-time GST filing rate",
    "gst_vs_bank_turnover_gap_pct": "GST vs. bank turnover consistency",
    "months_since_last_gst_filing": "Recency of last GST filing",
    "gst_registration_vintage_years": "Years since GST registration",
    "employee_count": "Employee headcount",
    "employee_retention_rate_pct": "Employee retention rate",
    "pf_contribution_regularity_pct": "PF contribution regularity",
    "avg_salary_payment_delay_days": "Salary payment punctuality",
    "revenue_6m_growth_pct": "6-month revenue growth",
    "revenue_volatility_pct": "Revenue stability",
}

PILLAR_LABELS = {
    "cash_flow": "Cash Flow Health", "compliance": "Compliance Health",
    "stability": "Stability Health", "growth": "Growth Trend",
}


def normalize_feature(series, direction):
    lo, hi = series.quantile(0.05), series.quantile(0.95)
    if hi == lo:
        return pd.Series(50.0, index=series.index)
    clipped = series.clip(lo, hi)
    scaled = (clipped - lo) / (hi - lo) * 100
    if direction == "lower":
        scaled = 100 - scaled
    return scaled


def compute_data_maturity_factor(df):
    vintage = np.minimum(df["business_vintage_years"], df["gst_registration_vintage_years"])
    return ((vintage - MATURITY_LOW_YEARS) / (MATURITY_HIGH_YEARS - MATURITY_LOW_YEARS)).clip(0, 1)


def main():
    df = pd.read_csv(IN_PATH)
    maturity = compute_data_maturity_factor(df)
    df["data_maturity_factor"] = maturity.round(2)

    # Freed compliance weight redistributed to cash_flow / growth (per business)
    max_shrink = 0.5
    compliance_weight = BASE_PILLAR_WEIGHTS["compliance"] * (1 - max_shrink * (1 - maturity))
    freed = BASE_PILLAR_WEIGHTS["compliance"] - compliance_weight
    pillar_weights = pd.DataFrame({
        "cash_flow": BASE_PILLAR_WEIGHTS["cash_flow"] + freed * PILLAR_REDISTRIBUTION_SPLIT["cash_flow"],
        "compliance": compliance_weight,
        "stability": BASE_PILLAR_WEIGHTS["stability"],
        "growth": BASE_PILLAR_WEIGHTS["growth"] + freed * PILLAR_REDISTRIBUTION_SPLIT["growth"],
    })

    # Normalize every raw feature once; store per-feature normalized value + its
    # TOTAL effective weight toward the composite score (within-pillar weight x pillar weight)
    normalized = {}
    total_weight = {}
    for pillar, features in PILLAR_FEATURES.items():
        vintage_dep = [(c, d, w) for c, d, w, vd in features if vd]
        behavior = [(c, d, w) for c, d, w, vd in features if not vd]
        behavior_base_weight = sum(w for _, _, w in behavior) if behavior else 0
        vintage_base_weight = sum(w for _, _, w in vintage_dep) if vintage_dep else 0

        for col, direction, base_w in behavior:
            norm = normalize_feature(df[col], direction)
            normalized[col] = norm
            if vintage_dep:
                scale_factor = 1 + ((vintage_base_weight * (1 - maturity)) / behavior_base_weight)
            else:
                scale_factor = 1.0
            eff_within_pillar = base_w * scale_factor
            total_weight[col] = eff_within_pillar * pillar_weights[pillar]

        for col, direction, base_w in vintage_dep:
            norm = normalize_feature(df[col], direction)
            normalized[col] = norm
            eff_within_pillar = base_w * maturity
            total_weight[col] = eff_within_pillar * pillar_weights[pillar]

    norm_df = pd.DataFrame(normalized)
    weight_df = pd.DataFrame(total_weight)

    # Pillar scores + composite (same as adaptive engine) for verification
    for pillar, features in PILLAR_FEATURES.items():
        cols = [c for c, *_ in features]
        pillar_within_weights = weight_df[cols].div(pillar_weights[pillar], axis=0)
        df[f"score_{pillar}"] = (norm_df[cols] * pillar_within_weights).sum(axis=1).round(1)

    composite_0_100 = sum(df[f"score_{pillar}"] * pillar_weights[pillar] for pillar in PILLAR_FEATURES)
    df["composite_score_0_100"] = composite_0_100.round(1)
    df["financial_health_score"] = (300 + (composite_0_100 / 100) * 600).round(0).astype(int)
    df["risk_band"] = pd.cut(
        df["financial_health_score"], bins=[-np.inf, 549, 649, 749, np.inf],
        labels=["High Risk", "Moderate", "Good", "Excellent"]
    )

    df.to_csv(OUT_SCORED_PATH, index=False)

    # ------------------- Explainability: exact point-contribution per feature -------------------
    all_features = [c for feats in PILLAR_FEATURES.values() for c, *_ in feats]
    population_avg = {c: norm_df[c].mean() for c in all_features}

    explanations = {}
    for idx in df.index:
        contributions = []
        for c in all_features:
            deviation = norm_df.loc[idx, c] - population_avg[c]
            points = round(deviation * weight_df.loc[idx, c] * 6, 1)
            contributions.append({
                "feature": c,
                "label": FEATURE_LABELS[c],
                "points": points,
                "business_value": round(df.loc[idx, c], 2) if c in df.columns else None,
            })
        contributions.sort(key=lambda x: x["points"], reverse=True)
        top_positive = [c for c in contributions if c["points"] > 0][:3]
        top_negative = sorted([c for c in contributions if c["points"] < 0], key=lambda x: x["points"])[:3]

        bid = df.loc[idx, "business_id"]
        score = int(df.loc[idx, "financial_health_score"])
        band = str(df.loc[idx, "risk_band"])

        narrative_parts = [f"{bid} scores {score} ({band})."]
        if top_positive:
            best = top_positive[0]
            narrative_parts.append(f"Strongest driver: {best['label']} (+{best['points']} pts).")
        if top_negative:
            worst = top_negative[0]
            narrative_parts.append(f"Biggest drag: {worst['label']} ({worst['points']} pts).")
        if df.loc[idx, "data_maturity_factor"] < 0.5:
            narrative_parts.append(
                "Note: thin credit history — compliance pillar weight reduced and "
                "reallocated to cash flow/growth to avoid penalizing this business for its age."
            )

        explanations[bid] = {
            "financial_health_score": score,
            "risk_band": band,
            "data_maturity_factor": round(float(df.loc[idx, "data_maturity_factor"]), 2),
            "pillar_scores": {
                PILLAR_LABELS[p]: round(float(df.loc[idx, f"score_{p}"]), 1) for p in PILLAR_FEATURES
            },
            "top_positive_drivers": top_positive,
            "top_negative_drivers": top_negative,
            "narrative": " ".join(narrative_parts),
        }

    with open(OUT_EXPLANATIONS_PATH, "w") as f:
        json.dump(explanations, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.integer, np.floating)) else str(o))

    # ------------------- Print a few examples -------------------
    print("=" * 70)
    print("EXPLAINABILITY EXAMPLES")
    print("=" * 70)
    sample_ids = df["business_id"].sample(3, random_state=1).tolist()
    # Force-include the NTC/NTB example we highlighted earlier if present
    for forced_id in ["MSME0044", "MSME0002"]:
        if forced_id in explanations and forced_id not in sample_ids:
            sample_ids.append(forced_id)

    for bid in sample_ids:
        exp = explanations[bid]
        print(f"\n--- {bid} ---")
        print(f"Score: {exp['financial_health_score']} ({exp['risk_band']}) | "
              f"Data maturity: {exp['data_maturity_factor']}")
        print(f"Pillar scores: {exp['pillar_scores']}")
        print("Top positive drivers:")
        for d in exp["top_positive_drivers"]:
            print(f"    + {d['label']}: +{d['points']} pts (value={d['business_value']})")
        print("Top negative drivers:")
        for d in exp["top_negative_drivers"]:
            print(f"    - {d['label']}: {d['points']} pts (value={d['business_value']})")
        print(f"Narrative: {exp['narrative']}")

    print(f"\nFull scored dataset -> {OUT_SCORED_PATH}")
    print(f"Full explanations JSON -> {OUT_EXPLANATIONS_PATH}")


if __name__ == "__main__":
    main()
