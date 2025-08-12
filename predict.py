import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from preprocessing.clean_data import clean_data

# Load saved model & label encoder
pipeline = joblib.load("model/xgb_pipeline.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# Load test data
test_df = pd.read_csv("data/test1.csv")  # change path

# Target column name
target_column = "Career Path"

# Apply the same cleaning as training
test_df = clean_data(test_df, target_column)

# Normalize label casing
if target_column in test_df.columns:
    test_df[target_column] = test_df[target_column].str.strip().str.title()

# Merge rare or unseen categories into 'Other'
known_classes = set(label_encoder.classes_)
test_df[target_column] = test_df[target_column].apply(
    lambda x: x if pd.notna(x) and x in known_classes else "Other"
)

# Combine text columns
test_df["combined_text"] = (
    test_df["Skills"].astype(str) + " " +
    test_df["Subjects"].astype(str) + " " +
    test_df["Interests"].astype(str)
)

# If ground truth exists
if target_column in test_df.columns:
    eval_df = test_df.dropna(subset=[target_column])

    if not eval_df.empty:
        y_true_encoded = label_encoder.transform(eval_df[target_column])
        y_pred_encoded = pipeline.predict(eval_df["combined_text"])

        y_true = label_encoder.inverse_transform(y_true_encoded)
        y_pred = label_encoder.inverse_transform(y_pred_encoded)

        # Accuracy and report
        accuracy = accuracy_score(y_true, y_pred)
        print(f"✅ Accuracy: {accuracy:.2f}")
        print("\n📊 Classification Report:\n", classification_report(y_true, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=label_encoder.classes_)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

        # Prediction Summary Bar Plot
        pred_counts = pd.Series(y_pred).value_counts()
        plt.figure(figsize=(8, 5))
        sns.barplot(x=pred_counts.index, y=pred_counts.values, palette="viridis")
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        plt.title("Prediction Distribution")
        plt.show()

# Save predictions
test_df["Predicted Career Path"] = pipeline.predict(test_df["combined_text"])
test_df["Predicted Career Path"] = label_encoder.inverse_transform(test_df["Predicted Career Path"])
test_df.to_csv("data/test1_output.csv", index=False)
print("\n💾 Predictions saved to data/test1_output.csv")
