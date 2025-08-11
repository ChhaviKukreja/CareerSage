import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
from preprocessing.clean_data import clean_data

# Load saved model & label encoder
pipeline = joblib.load("model/xgb_pipeline.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# Load test data
test_df = pd.read_csv("data/test.csv")  # change path

# Target column name
target_column = "Career Path"

# Apply the same cleaning as training
test_df = clean_data(test_df, target_column)

# Merge rare categories into 'Other' (same as training)
min_count = 2
class_counts = test_df[target_column].value_counts()
rare_classes = class_counts[class_counts < min_count].index
test_df[target_column] = test_df[target_column].replace(rare_classes, "Other")

# Combine text columns
test_df["combined_text"] = (
    test_df["Skills"].astype(str) + " " +
    test_df["Subjects"].astype(str) + " " +
    test_df["Interests"].astype(str)
)

# If ground truth exists
# If ground truth exists
if target_column in test_df.columns:
    # Separate rows with known labels for evaluation
    eval_df = test_df.dropna(subset=[target_column])

    if not eval_df.empty:
        # Encode labels
        y_true_encoded = label_encoder.transform(eval_df[target_column])
        y_pred_encoded = pipeline.predict(eval_df["combined_text"])

        # Decode to strings
        y_true = label_encoder.inverse_transform(y_true_encoded)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)

        # Accuracy and report
        print("✅ Accuracy:", accuracy_score(y_true, y_pred))
        print("\n📊 Classification Report:\n", classification_report(y_true, y_pred))

        # Summary of predictions
        print("\n🔍 Prediction Summary:")
        print(pd.Series(y_pred).value_counts())

# Save predictions to CSV
test_df.to_csv("data/noisy_test_predictions.csv", index=False)
print("\n💾 Predictions saved to data/test_output.csv")
