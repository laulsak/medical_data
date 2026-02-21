This project builds a machine learning model to predict whether a patient will be readmitted within 30 days after discharge using a synthetic healthcare dataset.

# Machine learning task:
-Type: Binary classification
-Target variable: readmitted_30d

Results (5-fold CV ROC-AUC):

**Logistic Regression (balanced)**
- Mean ROC-AUC: 0.724
- Std: 0.020

**Random Forest (balanced_subsample)**
- Mean ROC-AUC: 0.695
- Std: 0.030

**Hold-out Test ROC-AUC:** approx 0.699