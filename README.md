This project builds a machine learning model to predict whether a patient will be readmitted within 30 days after discharge using a synthetic healthcare dataset. It's a binary classification task focused on improving recall to minimize false negatives (missed readmissions), which can be costly in healthcare.


## Project Overview
- **Problem**: Binary classification to predict whether patient is readmitted within 30 days back to hospital. (0: No, 1: Yes).

- **Dataset**: Synthetic data with 3000 samples (approx 23% positive class, imbalanced). Features include age, gender, chronic conditions, admission type, etc.
 
- **Goal**: To build a reproducible pipeline for training and prediction, emphasizing stability and handling the imbalances

- **Key Insights**: Linear models like logistic regression perform well. Class weighting improves recall without sacrificing much accuracy.

## Methods
- **Preprocessing**: Used 'ColumnTransformer' and 'Pipeline' for one-hot encoding categorical features (for example gender and department) and passthrough for numerics. Prevents data leakage.

- **Models Compared**: 
    -Logistic regression (baseline, with 'class_weight="balanced"' for imbalance).
    -Random Forest (for non-linearity check).

- **Evaluation**: 5-fold cross-validation with ROC-AUC (robust for imbalanced data). Hold-out test set (10%) for final metrics.

- **Handling Imbalance**: 'class_weight="balanced"' prioritizes minority class, improving recall for readmissions.


## Results
- **5-fold CV ROC-AUC** :
    - Logistic Regression: Mean 0.724 (Std 0.020)
    - Random Forest: Mean 0.695 (Std 0.030)

- **Hold-out Test ROC-AUC**: approx 0.699 (Logistic Regression)

- **Classification Report (Test Set, Threshold=0.5)**:
    -Class 0: Precision 0.89, Recall 0.58, F1-score 0.71, Support 230
    -Class 1: Precision 0.36, Recall 0.77, F1-score 0.49, Support 70
    -Accuracy: 0.63 (Total support 300)
    -Macro Avg: Precision 0.63, Recall 0.68, F1-score 0.60, Support 300
    -Weighted Avg: Precision 0.77, Recall 0.63, F1-score 0.66, Support 300
- **Interpretation**: Good generalization (low std). Recall for class 1 (readmitted) is 0.77 - improved by class weighting, prioritizing missed readmissions in healthcare. Accuracy 0.63 is reasonable for imbalanced data

## How to run
- **Install Dependencies**: `pip install -r requirements.txt`
- **Train Model**: `python src/train.py --data data/healthcare_patient_journey.csv --model_out models/baseline_logreg.joblib --metrics_out models/metrics.json`
    -Outputs: Trained model, CV scores, test metrics.

- **Predict**: `python src/predict.py --model models/baseline_logreg.joblib --input data/new_patients.csv --output data/predictions.csv --threshold 0.5`

    -Outputs: CSV with predictions and probabilities.

## Future Work
- Threshold tuning for better recall (e.g., precision-recall curve).
- Feature importance analysis (e.g., coefficients from Logistic Regression).
- Try advanced models like Gradient Boosting (inspired by ISLR Chapter 8).


## References
- Dataset: Synthetic, inspired by real healthcare challenges
- Methods: Based on scikit-learn; concepts from ISLR By James et al.




