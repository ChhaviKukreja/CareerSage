import pandas as pd
import joblib
from collections import Counter
from preprocessing import engineer_features  # ✅ new pipeline

# 1️⃣ Load trained model and feature columns
model = joblib.load("models/career_model.joblib")
feature_columns = joblib.load("models/feature_columns.joblib")

# 2️⃣ Load raw test data
test_df = pd.read_csv("data/test_data.csv")

# 3️⃣ Feature engineering (training=False for inference)
X_test, _ = engineer_features(test_df, training=False)

# 4️⃣ Align columns to match training
for col in feature_columns:
    if col not in X_test.columns:
        X_test[col] = 0  # add missing features as 0
X_test = X_test[feature_columns].fillna(0)

# 5️⃣ Predict
preds = model.predict(X_test)

# 6️⃣ Stats on predictions
pred_counts = Counter(preds)
total_preds = len(preds)
other_count = pred_counts.get("Other", 0)

if other_count > 0:
    print(f"⚠️ Warning: {other_count} predictions are 'Other' — merged rare classes in training.")

# 7️⃣ Detailed prediction list
print("\nPredicted career paths:")
for i, pred in enumerate(preds, start=1):
    print(f"{i}. {pred}")

# 8️⃣ Summary stats
print("\n📊 Prediction Summary:")
summary_df = pd.DataFrame([
    {"Career Path": label, "Count": count, "Percentage": f"{(count/total_preds)*100:.2f}%"}
    for label, count in pred_counts.items()
]).sort_values(by="Count", ascending=False)

print(summary_df.to_string(index=False))
