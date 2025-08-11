from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print classification report."""
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
