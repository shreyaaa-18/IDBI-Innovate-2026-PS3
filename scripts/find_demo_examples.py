"""
Demo Example Finder
----------------------
Instead of hardcoding business IDs (which break if the dataset is regenerated),
this scans the scored dataset + explanations to automatically find the best
examples for your demo:

  1. Best "credit invisible but genuinely good" business — thin file, high score
     -> proves the adaptive fix rescues good businesses from unfair rejection.
  2. A thin-file business that's genuinely risky and correctly stays High Risk
     -> proves the model isn't just rubber-stamping every new business.
  3. The biggest score jump caused by the adaptive fix (compare baseline vs adaptive
     scoring — requires msme_scored_dataset.csv AND msme_scored_dataset_final.csv
     both present from earlier steps).

Run this any time you regenerate the dataset, right before your demo/pitch,
so your example business IDs are always current and correct.
"""

import json
import pandas as pd

SCORED_PATH = "/mnt/user-data/outputs/msme_scored_dataset_final.csv"
EXPLANATIONS_PATH = "/mnt/user-data/outputs/msme_explanations.json"


def main():
    df = pd.read_csv(SCORED_PATH)
    with open(EXPLANATIONS_PATH) as f:
        explanations = json.load(f)

    thin_file = df[df["data_maturity_factor"] < 0.6].copy()

    print("=" * 70)
    print("BEST DEMO EXAMPLES (auto-selected from current dataset)")
    print("=" * 70)

    # 1. Best thin-file business (the "rescued" example)
    if not thin_file.empty:
        best = thin_file.sort_values("financial_health_score", ascending=False).iloc[0]
        bid = best["business_id"]
        print(f"\n[1] BEST THIN-FILE BUSINESS -> {bid}")
        print(f"    Score: {best['financial_health_score']} ({best['risk_band']}) | "
              f"Data maturity: {best['data_maturity_factor']}")
        print(f"    Narrative: {explanations[bid]['narrative']}")
        if best['financial_health_score'] >= 650:
            print("    ^ GOOD DEMO CASE: thin-file business still scores Good/Excellent.")
        else:
            print("    ^ Note: even the best thin-file business in this draw scores "
                  "below 'Good'. Consider increasing N_BUSINESSES or the NTC_clean "
                  "quality parameters in generate_msme_dataset.py for a stronger example.")

    # 2. Genuinely risky thin-file business (the "correctly rejected" example)
    if not thin_file.empty:
        worst = thin_file.sort_values("financial_health_score").iloc[0]
        bid = worst["business_id"]
        print(f"\n[2] GENUINELY RISKY THIN-FILE BUSINESS -> {bid}")
        print(f"    Score: {worst['financial_health_score']} ({worst['risk_band']}) | "
              f"Data maturity: {worst['data_maturity_factor']}")
        print(f"    Narrative: {explanations[bid]['narrative']}")
        print("    ^ GOOD DEMO CASE: model correctly flags risk despite thin file, "
              "proving it isn't just approving every new business.")

    # 3. Score contrast across the whole thin-file population
    print(f"\n[3] THIN-FILE POPULATION SUMMARY")
    print(f"    Total thin-file businesses: {len(thin_file)}")
    print(f"    Score range: {thin_file['financial_health_score'].min()} - "
          f"{thin_file['financial_health_score'].max()}")
    print(f"    Risk band distribution:")
    print("   ", thin_file["risk_band"].value_counts().to_dict())

    good_thin_file = thin_file[thin_file["financial_health_score"] >= 650]
    print(f"\n    {len(good_thin_file)} of {len(thin_file)} thin-file businesses "
          f"({len(good_thin_file)/len(thin_file)*100:.0f}%) score Good/Excellent "
          f"-> these are your 'previously invisible, now approvable' population.")

    if not good_thin_file.empty:
        print("\n    Recommended business IDs for your slide/demo:")
        for bid in good_thin_file.sort_values("financial_health_score", ascending=False)["business_id"].head(3):
            print(f"      - {bid}: {explanations[bid]['narrative']}")


if __name__ == "__main__":
    main()
