import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import engineer_features  # ✅ use our updated feature engineering

def train_model():
    # 1️⃣ Load raw training data
    df_raw = pd.read_csv("data/improved_train.csv")  # raw input

    # 2️⃣ Feature engineering (training mode = True)
    X, y = engineer_features(df_raw, training=True)

    # 3️⃣ Fill missing values (basic handling)
    X = X.fillna(0)

    # 4️⃣ Merge rare classes into "Other"
    min_samples = 5
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < min_samples].index
    y = y.apply(lambda c: "Other" if c in rare_classes else c)

    # 5️⃣ Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6️⃣ Train temporary RF to find top features
    temp_model = RandomForestClassifier(random_state=42, n_estimators=200)
    temp_model.fit(X_train, y_train)

    importances = temp_model.feature_importances_
    top_k = 100  # keep top 100 features
    top_features_idx = np.argsort(importances)[-top_k:]
    top_feature_names = X.columns[top_features_idx]

    # 7️⃣ Keep only top features
    X_train = X_train[top_feature_names]
    X_test = X_test[top_feature_names]

    # 8️⃣ Train final model
    model = RandomForestClassifier(random_state=42, n_estimators=300)
    model.fit(X_train, y_train)

    # 9️⃣ Save model & top features
    joblib.dump(model, "models/career_model.joblib")
    joblib.dump(list(top_feature_names), "models/feature_columns.joblib")

    print(f"✅ Model trained and saved. Using {len(top_feature_names)} features.")

    # 🔟 Validation
    y_pred = model.predict(X_test)
    print("\n📊 Validation Results:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Accuracy:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    train_model()
