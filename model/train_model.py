from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def train_and_evaluate(X_train, y_train, X_test, y_test, label_encoder):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Decode labels back to original strings
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    print("✅ Model trained and evaluated.")
    print(classification_report(y_test_labels, y_pred_labels))
