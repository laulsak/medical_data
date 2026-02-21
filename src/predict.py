import argparse
from pathlib import Path

import joblib
import pandas as pd

DROP_COLS = ["patient_id", "readmitted_30d", "total_cost_€", "satisfaction_score"]

def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference using a trained model.")
    parser.add_argument("--model", default="models/baseline_logreg.joblib", help="Path to saved joblib model")
    parser.add_argument("--input", required=True, help="CSV file containing raw rows to score")
    parser.add_argument("--output", default="data/predictions.csv", help="Where to save predictions CSV")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for class 1")
    args = parser.parse_args()

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)


    model = joblib.load(model_path)

    df = pd.read_csv(input_path)

    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= args.threshold).astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({"pred": pred, "proba": proba})
    out_df.to_csv(output_path, index=False)


    print(f"Saved {len(out_df)} predictions to: {output_path}")


if __name__ == "__main__":
    main()

