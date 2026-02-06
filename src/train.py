import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("data/healthcare_patient_journey.csv")

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


