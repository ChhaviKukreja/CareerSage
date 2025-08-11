import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import load_data, clean_data, normalize_gpa, engineer_features

def train_model():
    df = load_data("data/noisy_student_dataset.csv")
    df = clean_data(df)
    df = normalize_gpa(df, training=True)
    X, y = engineer_features(df, training=True)

    X = X.fillna(0)
    rare_classes = y.value_counts()[lambda c: c < 5].index
    y = y.apply(lambda c: "Other" if c in rare_classes else c)

    mask = y.notna()
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    temp_model = RandomForestClassifier(random_state=42, n_estimators=200)
    temp_model.fit(X_train, y_train)
    top_features_idx = np.argsort(temp_model.feature_importances_)[-100:]
    top_feature_names = X.columns[top_features_idx]

    X_train, X_test = X_train[top_feature_names], X_test[top_feature_names]

    model = RandomForestClassifier(random_state=42, n_estimators=300)
    model.fit(X_train, y_train)

    joblib.dump(model, "models/career_model.joblib")
    joblib.dump(list(top_feature_names), "models/feature_columns.joblib")

    print("✅ Model trained and saved.")
    print(classification_report(y_test, model.predict(X_test)))

if __name__ == "__main__":
    train_model()
