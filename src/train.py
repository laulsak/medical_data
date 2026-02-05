import pandas as pd


df = pd.read_csv("data/healthcare_patient_journey.csv")

y = df["readmitted_30d"]
x = df.drop(columns=["patient_id", "readmitted_30d", "total_cost_€", "satisfaction_score"])

print("X shape: ", x.shape)
print("y distribution ", y.value_counts(normalize=True))
