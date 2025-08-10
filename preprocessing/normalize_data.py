from sklearn.preprocessing import MinMaxScaler

def normalize_gpa(df):
    scaler = MinMaxScaler()
    df['GPA'] = scaler.fit_transform(df[['GPA']])
    return df
