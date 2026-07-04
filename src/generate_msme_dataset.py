"""
Synthetic MSME Financial Health Dataset Generator
---------------------------------------------------
Generates realistic-looking MSME profiles across 4 data pillars:
  1. Cash Flow Health   (UPI / bank transaction behavior)
  2. Compliance Health  (GST filing behavior)
  3. Stability Health   (EPFO / employee data)
  4. Growth Trend       (revenue trajectory)

Each business is assigned a hidden "true_segment" (Healthy / Average / Risky / NTC-NTB)
which drives correlated feature generation, so the data has real signal for your
scoring model and ML classifier to pick up on (not pure noise).

Output: msme_synthetic_dataset.csv
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_BUSINESSES = 150

np.random.seed(RNG_SEED)

SECTORS = [
    "Retail Trade", "Food & Beverage", "Textile & Apparel", "Manufacturing (Light)",
    "IT Services", "Auto Repair & Parts", "Construction Materials", "Pharma & Healthcare Retail",
    "Agri-Trading", "Logistics & Transport"
]

CITIES_TIER = ["Tier-1", "Tier-2", "Tier-3"]

# Segment distribution: reflects real-world skew — most MSMEs are "Average" or "NTC/NTB"
SEGMENTS = np.random.choice(
    ["Healthy", "Average", "Risky", "NTC_NTB"],
    size=N_BUSINESSES,
    p=[0.20, 0.35, 0.20, 0.25]
)

# Segment -> underlying quality parameters (mean, std) used to draw correlated features.
# NTC_NTB = New-to-Credit / New-to-Bank: thin transaction history, but not necessarily risky.
SEGMENT_PROFILE = {
    "Healthy":  dict(cashflow_mu=85, compliance_mu=88, stability_mu=82, growth_mu=12, vintage_mu=7, noise=6),
    "Average":  dict(cashflow_mu=62, compliance_mu=60, stability_mu=58, growth_mu=4,  vintage_mu=4, noise=9),
    "Risky":    dict(cashflow_mu=35, compliance_mu=32, stability_mu=30, growth_mu=-6, vintage_mu=3, noise=10),
    "NTC_NTB":  dict(cashflow_mu=55, compliance_mu=40, stability_mu=50, growth_mu=6,  vintage_mu=0.8, noise=12),
}


def clip(x, lo=0, hi=100):
    return np.clip(x, lo, hi)


def generate_business(i, segment):
    p = SEGMENT_PROFILE[segment]
    noise = p["noise"]

    # ---------- Pillar 1: Cash Flow Health (from UPI / bank transactions) ----------
    monthly_upi_volume = max(5000, np.random.normal(p["cashflow_mu"] * 800, 15000))       # ₹ per month
    avg_monthly_balance = max(2000, np.random.normal(p["cashflow_mu"] * 500, 8000))         # ₹
    txn_count_per_month = max(10, np.random.normal(p["cashflow_mu"] * 3, 25))
    bounce_rate_pct = clip(np.random.normal(100 - p["cashflow_mu"], noise) * 0.15, 0, 25)    # % of txns bounced/failed
    balance_volatility_pct = clip(np.random.normal(100 - p["cashflow_mu"], noise) * 0.4, 5, 90)  # coefficient of variation proxy

    # ---------- Pillar 2: Compliance Health (GST filing behavior) ----------
    gst_on_time_filing_pct = clip(np.random.normal(p["compliance_mu"], noise))
    gst_vs_bank_turnover_gap_pct = clip(np.random.normal(100 - p["compliance_mu"], noise) * 0.5, 0, 80)  # mismatch %
    months_since_last_gst_filing = max(0, round(np.random.normal((100 - p["compliance_mu"]) / 25, 1)))
    gst_registration_vintage_years = max(0.1, np.random.normal(p["vintage_mu"], 2))

    # ---------- Pillar 3: Stability Health (EPFO / employee data) ----------
    employee_count = max(1, round(np.random.normal(p["stability_mu"] / 8, 5)))
    employee_retention_rate_pct = clip(np.random.normal(p["stability_mu"], noise))
    pf_contribution_regularity_pct = clip(np.random.normal(p["stability_mu"], noise))
    avg_salary_payment_delay_days = max(0, np.random.normal((100 - p["stability_mu"]) / 4, 5))

    # ---------- Pillar 4: Growth Trend ----------
    revenue_6m_growth_pct = np.random.normal(p["growth_mu"], noise * 0.8)
    revenue_volatility_pct = clip(np.random.normal(100 - p["cashflow_mu"], noise) * 0.3, 5, 70)

    # ---------- Business vintage & segment context ----------
    business_vintage_years = max(0.1, np.random.normal(p["vintage_mu"] + 1, 2))
    is_ntc = segment == "NTC_NTB"

    # ---------- Hidden "ground truth" default label for ML training/validation ----------
    # Higher risk of default correlates inversely with the average of the 4 pillar means.
    risk_index = 100 - np.mean([p["cashflow_mu"], p["compliance_mu"], p["stability_mu"]]) + np.random.normal(0, 8)
    default_probability = clip(risk_index, 2, 95) / 100
    defaulted_12m = np.random.binomial(1, default_probability * 0.5)  # dampened base rate, realistic ~10-15% overall

    return {
        "business_id": f"MSME{i+1:04d}",
        "business_name": f"{segment.split('_')[0]} Enterprises {i+1}",
        "sector": np.random.choice(SECTORS),
        "city_tier": np.random.choice(CITIES_TIER, p=[0.35, 0.4, 0.25]),
        "business_vintage_years": round(business_vintage_years, 1),
        "segment_hidden_label": segment,      # hidden ground truth — don't feed to scoring model directly
        "is_ntc_ntb": is_ntc,

        # Cash Flow pillar
        "monthly_upi_volume_inr": round(monthly_upi_volume, 0),
        "avg_monthly_balance_inr": round(avg_monthly_balance, 0),
        "txn_count_per_month": round(txn_count_per_month, 0),
        "bounce_rate_pct": round(bounce_rate_pct, 2),
        "balance_volatility_pct": round(balance_volatility_pct, 2),

        # Compliance pillar
        "gst_on_time_filing_pct": round(gst_on_time_filing_pct, 2),
        "gst_vs_bank_turnover_gap_pct": round(gst_vs_bank_turnover_gap_pct, 2),
        "months_since_last_gst_filing": int(months_since_last_gst_filing),
        "gst_registration_vintage_years": round(gst_registration_vintage_years, 1),

        # Stability pillar
        "employee_count": int(employee_count),
        "employee_retention_rate_pct": round(employee_retention_rate_pct, 2),
        "pf_contribution_regularity_pct": round(pf_contribution_regularity_pct, 2),
        "avg_salary_payment_delay_days": round(avg_salary_payment_delay_days, 1),

        # Growth pillar
        "revenue_6m_growth_pct": round(revenue_6m_growth_pct, 2),
        "revenue_volatility_pct": round(revenue_volatility_pct, 2),

        # ML training target (use only for model validation / default classifier, not for the score itself)
        "defaulted_12m": int(defaulted_12m),
    }


def main():
    records = [generate_business(i, seg) for i, seg in enumerate(SEGMENTS)]
    df = pd.DataFrame(records)

    out_path = "/mnt/user-data/outputs/msme_synthetic_dataset.csv"
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} synthetic MSME records -> {out_path}\n")
    print("Segment distribution:")
    print(df["segment_hidden_label"].value_counts(), "\n")
    print("Default rate by segment:")
    print(df.groupby("segment_hidden_label")["defaulted_12m"].mean().round(3), "\n")
    print("Sample rows:")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
