import pandas as pd

def clean_degrees(df):
    df['Education Level'] = df['Education Level'].str.replace(r'\.', '', regex=True)
    df['Education Level'] = df['Education Level'].str.strip().str.lower()
    return df

def clean_column_text(df, columns):
    for col in columns:
        df[col] = df[col].astype(str).str.strip().str.title()
    return df

def handle_missing_values(df):
    df = df.drop_duplicates()
    df['Name'] = df['Name'].fillna('Unknown')
    df['GPA'] = pd.to_numeric(df['GPA'], errors='coerce')
    df['GPA'] = df['GPA'].fillna(df['GPA'].mean())
    df = df.fillna('Unknown')
    return df

def clean_data(df):
    df = clean_degrees(df)
    df = clean_column_text(df, ['Name', 'Subjects', 'Career Path'])
    df = handle_missing_values(df)
    return df
