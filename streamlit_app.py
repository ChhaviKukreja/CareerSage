# streamlit_app.py
#ru using streamlit run streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ---- Config: adjust paths if needed ----
MODEL_PATH = "model/xgb_pipeline.pkl"
LABEL_ENCODER_PATH = "model/label_encoder.pkl"

# ---- Load model & encoder with friendly error messages ----
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load model at {MODEL_PATH}: {e}")
        return None, None

    try:
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    except Exception as e:
        st.error(f"Could not load label encoder at {LABEL_ENCODER_PATH}: {e}")
        return model, None

    return model, label_encoder

model, label_encoder = load_artifacts()
if model is None or label_encoder is None:
    st.stop()

# ---- Helper functions ----
def preprocess_input(education, gpa, skills_text, subjects_text, interests_text):
    # Combine fields into a single text string (same style as training)
    # Add education and GPA as tokens (simple approach — works when your model uses text only)
    parts = [
        str(education).strip(),
        f"gpa_{gpa}",
        str(skills_text).strip(),
        str(subjects_text).strip(),
        str(interests_text).strip(),
    ]
    combined = " ".join([p for p in parts if p and p != "nan"])
    # ensure type string (TfidfVectorizer will lowercase by default)
    return combined

def top_k_predictions(probs, classes, k=3):
    # probs: 1D array, classes: array of labels
    idx = np.argsort(probs)[::-1][:k]
    return [(classes[i], float(probs[i])) for i in idx]

# ---- UI ----
st.title("CaeerSage- Career Path Predictor")
st.write("Enter candidate details and get the predicted career path.")

with st.form("input_form"):
    # Education
    education = st.selectbox(
        "Education Level",
        options=[
            "Bachelors", "Masters", "Phd", "Diploma", "High School", "Other"
        ],
        index=0
    )

    # GPA slider — adjust range to match your dataset (0-10 or 0-4)
    gpa = st.slider("GPA (approx.)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)

    # Free text fields for skills/subjects/interests
    st.markdown("**Skills** (comma or newline separated)")
    skills_text = st.text_area("Skills (e.g. Python, SQL, Communication)", height=80)

    st.markdown("**Subjects** (comma separated)")
    subjects_text = st.text_area("Subjects (e.g. Math; Statistics)", height=60)

    st.markdown("**Interests** (comma separated)")
    interests_text = st.text_area("Interests (e.g. Research, Design)", height=60)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_text = preprocess_input(education, gpa, skills_text, subjects_text, interests_text)
    st.write("**Combined input:**")
    st.code(input_text)

    # predict
    try:
        pred_encoded = model.predict([input_text])
        pred_label = label_encoder.inverse_transform(pred_encoded)[0]
        st.success(f"Predicted Career Path: **{pred_label}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # probabilities (if model supports predict_proba)
    try:
        probs = model.predict_proba([input_text])[0]
        classes = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))
        top3 = top_k_predictions(probs, classes, k=5)

        st.write("**Top predictions (with probabilities)**")
        df_top = pd.DataFrame(top3, columns=["Career Path", "Probability"])
        df_top["Probability (%)"] = (df_top["Probability"] * 100).round(2)
        st.table(df_top)

        # simple bar chart
        st.write("Probability distribution (all classes)")
        df_all = pd.DataFrame({"prob": probs}, index(label_encoder.classes_))
        st.bar_chart(df_all["prob"])

    except Exception:
        # If predict_proba not supported, ignore gracefully
        st.info("Model does not expose `predict_proba()` — only single-label prediction shown.")

# batch upload predictions
st.markdown("---")
st.subheader("Batch predict from CSV")
st.write("Upload a CSV with columns `Skills`, `Subjects`, `Interests`, `Education` (optional), `GPA` (optional). We'll add `Predicted Career Path` column.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df_in = pd.read_csv(uploaded)
    # fill missing columns if required
    for c in ["Skills", "Subjects", "Interests", "Education", "GPA"]:
        if c not in df_in.columns:
            df_in[c] = ""

    # create combined_text
    df_in["combined_text"] = df_in.apply(
        lambda r: preprocess_input(
            r.get("Education", ""), r.get("GPA", ""), r.get("Skills", ""), r.get("Subjects", ""), r.get("Interests", "")
        ),
        axis=1
    )

    # predict
    preds_enc = model.predict(df_in["combined_text"].tolist())
    try:
        preds = label_encoder.inverse_transform(preds_enc)
    except Exception:
        preds = preds_enc  # if encoder missing, show encoded labels
    df_in["Predicted Career Path"] = preds

    st.write(df_in.head(50))
    st.download_button("Download predictions CSV", df_in.to_csv(index=False).encode("utf-8"), "predictions.csv")
