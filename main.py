"""
Customer Churn Prediction - Streamlit App
------------------------------------------
- Auto-detects "Churn_Modelling.csv" in the working directory (or lets you upload it).
- Trains the encoders / scaler / ANN model fully IN-MEMORY using
  st.cache_resource, so nothing is written to disk (no .pkl / .h5 / .keras
  files get saved in the folder). Everything lives in Streamlit's cache for
  the lifetime of the session/app process.
- If you already have model.h5 / model.keras / *.pkl files sitting next to
  this script, the app will use those instead of retraining (still loaded
  once and cached in memory, not re-saved).
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="centered")

CSV_NAME = "Churn_Modelling.csv"
MODEL_CANDIDATES = ["model.h5", "model.keras"]
SCALER_PATH = "scaler.pkl"
GENDER_ENC_PATH = "label_encoder_gender.pkl"
GEO_ENC_PATH = "onehot_encoder_geo.pkl"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def find_existing_csv():
    """Auto-pick Churn_Modelling.csv from the current working dir if present."""
    if os.path.exists(CSV_NAME):
        return CSV_NAME
    return None


def find_existing_artifacts():
    """Check whether pre-trained model + encoders already exist on disk."""
    model_path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
    have_all = (
        model_path is not None
        and os.path.exists(SCALER_PATH)
        and os.path.exists(GENDER_ENC_PATH)
        and os.path.exists(GEO_ENC_PATH)
    )
    return have_all, model_path


@st.cache_data(show_spinner=False)
def load_csv(file_bytes_or_path):
    if isinstance(file_bytes_or_path, str):
        return pd.read_csv(file_bytes_or_path)
    return pd.read_csv(file_bytes_or_path)


@st.cache_resource(show_spinner="Loading saved model + encoders...")
def load_saved_artifacts(model_path):
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    model = load_model(model_path)

    with open(GENDER_ENC_PATH, "rb") as f:
        gender_enc = pickle.load(f)
    with open(GEO_ENC_PATH, "rb") as f:
        geo_enc = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, gender_enc, geo_enc, scaler


@st.cache_resource(show_spinner="Training model")
def train_in_memory(_df):
    """
    Trains encoders, scaler, and the ANN purely in memory.
    Nothing is pickled or saved to disk here -- everything stays cached
    as Python objects via st.cache_resource for reuse across reruns.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.callbacks import EarlyStopping

    df = _df.copy()
    for col in ["RowNumber", "CustomerId", "Surname"]:
        if col in df.columns:
            df = df.drop(col, axis=1)

    gender_enc = LabelEncoder()
    df["Gender"] = gender_enc.fit_transform(df["Gender"])

    geo_enc = OneHotEncoder()
    geo_encoded = geo_enc.fit_transform(df[["Geography"]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded, columns=geo_enc.get_feature_names_out(["Geography"])
    )

    df = pd.concat(
        [df.drop("Geography", axis=1).reset_index(drop=True), geo_encoded_df],
        axis=1,
    )

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential(
        [
            Input(shape=(X_train_scaled.shape[1],)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=100,
        callbacks=[early_stop],
        verbose=0,
    )

    val_loss, val_acc = model.evaluate(X_test_scaled, y_test, verbose=0)

    return model, gender_enc, geo_enc, scaler, X.columns.tolist(), val_acc


def build_input_df(raw_input, gender_enc, geo_enc):
    geo_encoded = geo_enc.transform([[raw_input["Geography"]]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded, columns=geo_enc.get_feature_names_out(["Geography"])
    )

    input_df = pd.DataFrame([raw_input])
    input_df["Gender"] = gender_enc.transform(input_df["Gender"])
    input_df = pd.concat(
        [input_df.drop("Geography", axis=1).reset_index(drop=True), geo_encoded_df],
        axis=1,
    )
    return input_df


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("📉 Customer Churn Prediction")

has_artifacts, model_path = find_existing_artifacts()
auto_csv = find_existing_csv()

with st.sidebar:
    st.header("⚙️ Setup")
    if has_artifacts:
        st.success(f"Found saved model: `{model_path}` + encoders/scaler. Using these.")
    elif auto_csv:
        st.info(f"Auto-detected `{auto_csv}` — will train a fresh model.")

    if st.button("🗑️ Clear Cached Model/Encoders", use_container_width=True):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Cache cleared. Rerunning...")
        st.rerun()

    uploaded_csv = None
    if not has_artifacts and not auto_csv:
        uploaded_csv = st.file_uploader("Upload Churn_Modelling.csv", type=["csv"])

model = gender_enc = geo_enc = scaler = None
val_acc = None

if has_artifacts:
    model, gender_enc, geo_enc, scaler = load_saved_artifacts(model_path)
else:
    csv_source = auto_csv if auto_csv else uploaded_csv
    if csv_source is not None:
        df = load_csv(csv_source)
        with st.spinner("Preparing model"):
            model, gender_enc, geo_enc, scaler, feature_cols, val_acc = train_in_memory(df)
        st.sidebar.success(f"Model trained in memory. Validation accuracy: {val_acc:.2%}")
    else:
        st.error(
            f"Place **{CSV_NAME}** in this app's working directory (or upload it in the sidebar) "
        )
        st.stop()

st.divider()
st.subheader("🔍 Predict Churn for a Customer")

col1, col2 = st.columns(2)
with col1:
    geography = st.selectbox("Geography", options=list(geo_enc.categories_[0]))
    gender = st.selectbox("Gender", options=list(gender_enc.classes_))
    age = st.slider("Age", 18, 100, 35)
    credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=650)
    tenure = st.slider("Tenure (years)", 0, 10, 3)

with col2:
    balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=100.0)
    num_products = st.slider("Number of Products", 1, 4, 1)
    has_cr_card = st.selectbox("Has Credit Card?", options=["No", "Yes"])
    is_active = st.selectbox("Is Active Member?", options=["No", "Yes"])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0, step=100.0)

if st.button("Predict", type="primary", use_container_width=True):
    raw_input = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": estimated_salary,
    }

    input_df = build_input_df(raw_input, gender_enc, geo_enc)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled, verbose=0)
    prediction_proba = float(prediction[0][0])

    st.metric("Churn Probability", f"{prediction_proba:.2%}")
    st.progress(min(max(prediction_proba, 0.0), 1.0))

    if prediction_proba > 0.5:
        st.error("⚠️ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **not likely to churn**.")