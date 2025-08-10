import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocessing.feature_engineering import engineer_features

def train_model():
    # Load raw training data
    df = pd.read_csv("data/improved_train.csv")

    # Engineer features and labels
    X, y = engineer_features(df, training=True)

    # Fill NaNs
    X = X.fillna(0)

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    print("\n📊 Validation Results:")
    print(classification_report(y_val, y_pred))
    print("Accuracy:", accuracy_score(y_val, y_pred))

    # Save model and feature column order
    joblib.dump(model, "models/career_model.joblib")
    joblib.dump(list(X.columns), "models/feature_columns.joblib")

    print("\n✅ Model trained and saved.")
    print(f"✅ Feature columns saved: {len(X.columns)} columns")

if __name__ == "__main__":
    train_model()
