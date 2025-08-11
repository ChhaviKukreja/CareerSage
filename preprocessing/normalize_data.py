from sklearn.preprocessing import MinMaxScaler
import joblib
import os

ENCODERS_DIR = "models/encoders"
os.makedirs(ENCODERS_DIR, exist_ok=True)

def normalize_gpa(df, training=True):
    scaler_path = os.path.join(ENCODERS_DIR, "gpa_scaler.pkl")
    scaler = MinMaxScaler()

    if training:
        df['GPA'] = scaler.fit_transform(df[['GPA']])
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        df['GPA'] = scaler.transform(df[['GPA']])
    
    return df
