import numpy as np
import pandas as pd
import ast
import os
import joblib
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

ENCODERS_DIR = "models/encoders"
os.makedirs(ENCODERS_DIR, exist_ok=True)

def parse_list_column(df, column):
    """Safely parse list-like strings into Python lists."""
    def parse(x):
        if isinstance(x, list):
            return [str(i).strip() for i in x]
        if isinstance(x, str):
            try:
                val = ast.literal_eval(x)
                if isinstance(val, list):
                    return [str(i).strip() for i in val]
            except:
                return [i.strip() for i in x.split(';')]
        return []
    df[column] = df[column].apply(parse)
    return df

def engineer_features(df, training=True):
    df = df[['Education Level', 'GPA', 'Subjects', 'Skills', 'Interests', 'Career Path']].copy()

    df = parse_list_column(df, 'Subjects')
    df = parse_list_column(df, 'Skills')
    df = parse_list_column(df, 'Interests')

    # -----------------------------
    # Education Level Encoding
    # -----------------------------
    le_path = os.path.join(ENCODERS_DIR, "label_encoder_edu.pkl")
    le = LabelEncoder()

    if training:
        df['Education Level'] = le.fit_transform(df['Education Level'].astype(str))
        joblib.dump(le, le_path)
    else:
        le = joblib.load(le_path)
        df['Education Level'] = df['Education Level'].apply(
            lambda x: x if x in le.classes_ else "Other"
        )
        # Add "Other" if missing from training classes
        if "Other" not in le.classes_:
            le.classes_ = np.append(le.classes_, "Other")
        
        df['Education Level'] = le.transform(df['Education Level'])

    # -----------------------------
    # MultiLabelBinarizer for Subjects, Skills, Interests
    # -----------------------------
    mlb_paths = {
        "subjects": os.path.join(ENCODERS_DIR, "mlb_subjects.pkl"),
        "skills": os.path.join(ENCODERS_DIR, "mlb_skills.pkl"),
        "interests": os.path.join(ENCODERS_DIR, "mlb_interests.pkl"),
    }

    encoders = {
        "subjects": MultiLabelBinarizer(),
        "skills": MultiLabelBinarizer(),
        "interests": MultiLabelBinarizer(),
    }

    encoded_frames = []
    for col, key in zip(['Subjects', 'Skills', 'Interests'], ['subjects', 'skills', 'interests']):
        if training:
            # Add "Other" explicitly
            all_values = set(v for row in df[col] for v in row)
            all_values.add("Other")
            encoders[key].fit([list(all_values)])
            joblib.dump(encoders[key], mlb_paths[key])
        else:
            encoders[key] = joblib.load(mlb_paths[key])
            # Replace unseen values with "Other"
            df[col] = df[col].apply(lambda lst: [v if v in encoders[key].classes_ else "Other" for v in lst])

        encoded = pd.DataFrame(
            encoders[key].transform(df[col]),
            columns=[f"{col[:-1]}_{cls}" for cls in encoders[key].classes_]
        )
        encoded_frames.append(encoded)

    X = pd.concat([df[['Education Level', 'GPA']].reset_index(drop=True)] + encoded_frames, axis=1)
    y = df['Career Path']

    return X, y
