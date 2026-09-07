import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Earthquake Damage Prediction",
    page_icon="🏚️",
    layout="centered"
)


# ============================================================
# FREQUENCY ENCODER
# Harus sama dengan class saat training
# ============================================================

class FrequencyEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        self.freq_maps_ = {
            col: X[col].value_counts(normalize=True)
            for col in X.columns
        }

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col in X.columns:
            X[col] = (
                X[col]
                .map(self.freq_maps_[col])
                .fillna(0.0)
            )

        return X.values.astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array([
            f"{c}_freq"
            for c in self.freq_maps_.keys()
        ])


# ============================================================
# LOAD MODEL & METADATA
# ============================================================

MODEL_PATH = "earthquake_damage_model.joblib"
METADATA_PATH = "feature_metadata.json"


@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    return model


@st.cache_data
def load_metadata():

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
meta = load_metadata()


# ============================================================
# METADATA
# ============================================================

FEATURE_ORDER = meta["feature_order"]
CLASS_NAMES = meta["class_names"]

# ============================================================
# GET CATEGORIES DIRECTLY FROM MODEL
# ============================================================

preprocessor = model.named_steps["preprocessor"]

categorical_transformer = preprocessor.named_transformers_["categorical"]

encoder = categorical_transformer.named_steps["ohe"]

encoded_categories = encoder.categories_

categorical_columns = [
    "foundation_type",
    "ground_floor_type",
    "land_surface_condition",
    "legal_ownership_status",
    "other_floor_type",
    "plan_configuration",
    "position",
    "roof_type",
    "geo_level_1_id"
]

categorical = {
    col: list(cats)
    for col, cats in zip(categorical_columns, encoded_categories)
}


# ============================================================
# TITLE
# ============================================================

st.title("🏚️ Earthquake Building Damage Prediction")

st.write(
    "Aplikasi untuk memprediksi tingkat kerusakan bangunan "
    "berdasarkan karakteristik bangunan dan lokasi."
)

st.info(
    "Model: Random Forest dengan fitur struktural, penggunaan "
    "bangunan, dan karakteristik lokasi."
)


# ============================================================
# BUILDING LOCATION
# ============================================================

st.subheader("📍 Building Location")

geo_level_1_id = st.selectbox(
    "Geo Level 1 ID",
    options=categorical["geo_level_1_id"]
)

col1, col2 = st.columns(2)

with col1:

    geo_level_2_id = st.number_input(
        "Geo Level 2 ID",
        min_value=0,
        step=1,
        value=0
    )

with col2:

    geo_level_3_id = st.number_input(
        "Geo Level 3 ID",
        min_value=0,
        step=1,
        value=0
    )


# ============================================================
# BASIC BUILDING CHARACTERISTICS
# ============================================================

st.subheader("🏢 Building Characteristics")

count_families = st.number_input(
    "Number of Families",
    min_value=0,
    step=1,
    value=1
)


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

st.subheader("🏗️ Building Structure")


col1, col2 = st.columns(2)

with col1:

    foundation_type = st.selectbox(
        "Foundation Type",
        options=categorical["foundation_type"]
    )

    ground_floor_type = st.selectbox(
        "Ground Floor Type",
        options=categorical["ground_floor_type"]
    )

    land_surface_condition = st.selectbox(
        "Land Surface Condition",
        options=categorical["land_surface_condition"]
    )

    legal_ownership_status = st.selectbox(
        "Legal Ownership Status",
        options=categorical["legal_ownership_status"]
    )


with col2:

    other_floor_type = st.selectbox(
        "Other Floor Type",
        options=categorical["other_floor_type"]
    )

    plan_configuration = st.selectbox(
        "Plan Configuration",
        options=categorical["plan_configuration"]
    )

    position = st.selectbox(
        "Building Position",
        options=categorical["position"]
    )

    roof_type = st.selectbox(
        "Roof Type",
        options=categorical["roof_type"]
    )


# ============================================================
# SECONDARY USE
# ============================================================

st.subheader("🏪 Secondary Building Use")


has_secondary_use = st.checkbox(
    "Has Secondary Use?"
)

col1, col2 = st.columns(2)

with col1:

    has_secondary_use_agriculture = st.checkbox(
        "Agriculture"
    )

    has_secondary_use_gov_office = st.checkbox(
        "Government Office"
    )

    has_secondary_use_health_post = st.checkbox(
        "Health Post"
    )

    has_secondary_use_hotel = st.checkbox(
        "Hotel"
    )

    has_secondary_use_industry = st.checkbox(
        "Industry"
    )

    has_secondary_use_institution = st.checkbox(
        "Institution"
    )


with col2:

    has_secondary_use_other = st.checkbox(
        "Other"
    )

    has_secondary_use_rental = st.checkbox(
        "Rental"
    )

    has_secondary_use_school = st.checkbox(
        "School"
    )

    has_secondary_use_use_police = st.checkbox(
        "Police"
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Damage Level",
    use_container_width=True
)


if predict_button:

    # --------------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------------

    input_data = {

        "count_families":
            count_families,

        "foundation_type":
            foundation_type,

        "geo_level_1_id":
            geo_level_1_id,

        "geo_level_2_id":
            geo_level_2_id,

        "geo_level_3_id":
            geo_level_3_id,

        "ground_floor_type":
            ground_floor_type,

        "has_secondary_use":
            int(has_secondary_use),

        "has_secondary_use_agriculture":
            int(has_secondary_use_agriculture),

        "has_secondary_use_gov_office":
            int(has_secondary_use_gov_office),

        "has_secondary_use_health_post":
            int(has_secondary_use_health_post),

        "has_secondary_use_hotel":
            int(has_secondary_use_hotel),

        "has_secondary_use_industry":
            int(has_secondary_use_industry),

        "has_secondary_use_institution":
            int(has_secondary_use_institution),

        "has_secondary_use_other":
            int(has_secondary_use_other),

        "has_secondary_use_rental":
            int(has_secondary_use_rental),

        "has_secondary_use_school":
            int(has_secondary_use_school),

        "has_secondary_use_use_police":
            int(has_secondary_use_use_police),

        "land_surface_condition":
            land_surface_condition,

        "legal_ownership_status":
            legal_ownership_status,

        "other_floor_type":
            other_floor_type,

        "plan_configuration":
            plan_configuration,

        "position":
            position,

        "roof_type":
            roof_type
    }


    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    input_df = pd.DataFrame([input_data])

    # Pastikan urutan fitur sama dengan training
    input_df = input_df[FEATURE_ORDER]


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]


    # --------------------------------------------------------
    # HANDLE PREDICTION LABEL
    # --------------------------------------------------------

    if isinstance(prediction, (int, np.integer)):

        prediction_index = int(prediction)

        predicted_label = CLASS_NAMES[prediction_index]

    else:

        predicted_label = str(prediction)

        prediction_index = CLASS_NAMES.index(
            predicted_label
        )


    confidence = float(
        probabilities[prediction_index]
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.subheader("🎯 Prediction Result")


    if predicted_label == "Low":

        st.success(
            f"🏠 Predicted Damage Level: **{predicted_label}**"
        )

    elif predicted_label == "Medium":

        st.warning(
            f"⚠️ Predicted Damage Level: **{predicted_label}**"
        )

    else:

        st.error(
            f"🚨 Predicted Damage Level: **{predicted_label}**"
        )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2%}"
    )


    # ========================================================
    # PROBABILITY
    # ========================================================

    st.subheader("📊 Prediction Probability")


    probability_df = pd.DataFrame({

        "Damage Level":
            CLASS_NAMES,

        "Probability":
            probabilities

    })


    probability_df["Probability"] = (
        probability_df["Probability"] * 100
    )


    st.bar_chart(
        probability_df.set_index("Damage Level")
    )


    st.dataframe(
        probability_df.style.format({
            "Probability": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.caption(
        "⚠️ Hasil merupakan prediksi model berdasarkan pola "
        "pada dataset historis dan bukan pengganti kajian "
        "teknis atau inspeksi keselamatan bangunan."
    )
