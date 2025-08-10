import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import load_data, clean_data, normalize_gpa, engineer_features

# Load trained model & feature columns
model = joblib.load("models/career_model.joblib")
feature_columns = joblib.load("models/feature_columns.joblib")

# Load raw test data
test_df = load_data("data/test_data.csv")

# Preprocess
test_df = clean_data(test_df)
test_df = normalize_gpa(test_df)

# Engineer features
X_test, y_true = engineer_features(test_df, training=False)

# Align feature columns
for col in feature_columns:
    if col not in X_test.columns:
        X_test[col] = 0
X_test = X_test[feature_columns].fillna(0)

# Predict
y_pred = model.predict(X_test)

# Output predictions
print("\nPredicted career paths:", y_pred)

# If true labels exist, evaluate
if y_true is not None and not y_true.isnull().all():
    print("\n📊 Test Set Evaluation:")
    print(classification_report(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
else:
    print("\n⚠ No true labels found in test data. Only predictions shown.")
