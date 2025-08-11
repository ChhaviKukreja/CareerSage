import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def save_data(df, filepath):
    df.to_csv(filepath, index=False)
