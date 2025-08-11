import pandas as pd
import joblib

def run_predictions(test_file):
    # Load the saved model and label encoder
    model = joblib.load("model/xgb_pipeline.pkl")   # Adjust path if different
    label_encoder = joblib.load("model/label_encoder.pkl")

    # Load test data
    df = pd.read_csv(test_file)

    # Combine multiple text fields into one (same as training)
    df["combined_text"] = (
        df["skills"].astype(str) + " " +
        df["subjects"].astype(str) + " " +
        df["interests"].astype(str)
    )

    # Make predictions
    preds_encoded = model.predict(df["combined_text"])

    # Decode back to original class labels
    preds_labels = label_encoder.inverse_transform(preds_encoded)

    # Add predictions to DataFrame
    df["Predicted Career Path"] = preds_labels

    # Save predictions to CSV
    output_file = "predictions.csv"
    df.to_csv(output_file, index=False)

    print(f"Predictions saved to {output_file}")
    print(df[["skills", "subjects", "interests", "Predicted Career Path"]].head())

if __name__ == "__main__":
    run_predictions("test_data.csv")  # Change to your actual test file
