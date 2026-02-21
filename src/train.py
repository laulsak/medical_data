import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score
from pathlib import Path
import json
import joblib
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def cv_auc(model, x, y, name: str):
    scores = cross_val_score(
        model,
        x,
        y,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    print(f"\n{name} CV ROC-AUC scores: {scores}")
    print(f"{name} Mean CV ROC-AUC: {np.mean(scores)}")
    print(f"{name} Std CV ROC-AUC: {np.std(scores)}")

    return scores



def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/healthcare_patient_journey.csv")
    parser.add_argument("--model_out", default="models/baseline_logreg.joblib")
    parser.add_argument("--metrics_out", default="models/metrics.json")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    y = df["readmitted_30d"]
    x = df.drop(columns=["patient_id", "readmitted_30d", "total_cost_€", "satisfaction_score"])

    print("X shape: ", x.shape)
    print("y distribution ", y.value_counts(normalize=True))


    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42, stratify=y)


    categorical_cols = ["gender", "admission_type", "department", "discharge_status"]
    numeric_cols = [c for c in x.columns if c not in categorical_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    # Baseline model: Logistic regression
    clf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=500, class_weight="balanced"))
    ])

    rf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample"
        ))
    ])

    logit_scores = cv_auc(clf, x, y, "LogisticRegression")
    rf_scores = cv_auc(rf, x, y, "RandomForest")


        


    

   

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_proba = clf.predict_proba(x_test)[:, 1]

    print("\nROC-AUC:", roc_auc_score(y_test, y_proba))
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    model_path = Path(args.model_out)
    metrics_path = Path(args.metrics_out)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, model_path)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "test_size": 0.1,
        "random_state": 42,
        "n_samples": int(len(df)),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
                        



if __name__ == "__main__":
    main()














