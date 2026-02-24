import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Crop Yield Prediction System",
    page_icon="🌾",
    layout="wide"
)

# ----------------------------
# DARK MODE STYLING (Fixed sidebar, dropdown arrow, button)
# ----------------------------
st.markdown("""
<style>
/* App background & text */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #1f2b3a;
}

/* Sidebar labels, headers, text */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] span {
    color: white !important;
}

/* Selectbox input text */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #2c3e50 !important;
    color: white !important;
}

/* Make selectbox dropdown arrow visible */
.stSelectbox div[data-baseweb="select"] svg {
    fill: white !important;
}

/* Number input text */
.stNumberInput input {
    background-color: #2c3e50 !important;
    color: white !important;
}

/* Listbox dropdown options */
div[role="listbox"] {
    background-color: #2c3e50 !important;
    color: white !important;
}

/* Force Predict Yield button styling */
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] > button {
    background-color: #0b84a5 !important;   /* blue background */
    color: #ffffff !important;              /* white text */
    font-weight: bold !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 0.5em 1em !important;
    font-size: 0.95rem !important;
}

/* Hover state */
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] > button:hover {
    background-color: #09728f !important;
    color: #ffffff !important;
}

/* Disabled state */
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] > button:disabled {
    background-color: #0b84a5 !important;
    color: #ffffff !important;
    opacity: 0.8 !important;  /* slightly faded but readable */
}
}

/* Plotly charts background */
.plotly-graph-div {
    background-color: #141e30 !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL & DATA
# ----------------------------
model = pickle.load(open("model/crop_model.pkl", "rb"))
model_columns = pickle.load(open("model/model_columns.pkl", "rb"))

df_original = pd.read_csv("data/crop_data.csv")
df_original = df_original.dropna()
df_original["Yield"] = df_original["Production"] / df_original["Area"]

# ----------------------------
# TITLE
# ----------------------------
st.title("🌾 Crop Yield Prediction Dashboard")
st.write("Machine Learning Based Agricultural Yield Forecasting System")
st.write("---")

# ============================
# SIDEBAR FORM
# ============================
st.sidebar.header("📥 Enter Crop Details")

with st.sidebar.form("prediction_form"):

    crop_options = sorted([col.replace("Crop_", "") for col in model_columns if col.startswith("Crop_")])
    season_options = sorted([col.replace("Season_", "") for col in model_columns if col.startswith("Season_")])
    state_options = sorted([col.replace("State_Name_", "") for col in model_columns if col.startswith("State_Name_")])

    selected_crop = st.selectbox("Select Crop", crop_options)
    selected_season = st.selectbox("Select Season", season_options)
    selected_state = st.selectbox("Select State", state_options)
    area = st.number_input("Enter Cultivation Area", min_value=0.1, value=1.0)

    predict_button = st.form_submit_button("Predict Yield")

# ============================
# PREDICTION
# ============================
if predict_button:

    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0

    input_data["Area"] = area
    input_data[f"Crop_{selected_crop}"] = 1
    input_data[f"Season_{selected_season}"] = 1
    input_data[f"State_Name_{selected_state}"] = 1

    prediction = model.predict(input_data)[0]

    lower = prediction * 0.9
    upper = prediction * 1.1

    st.success(f"🌾 Predicted Yield: {round(prediction, 2)}")
    st.info(f"Estimated Range: {round(lower,2)} - {round(upper,2)}")

    # Crop-specific classification
    crop_df = df_original[df_original["Crop"] == selected_crop]

    st.subheader("📈 Yield Insight")

    if not crop_df.empty:
        low_thresh = crop_df["Yield"].quantile(0.33)
        high_thresh = crop_df["Yield"].quantile(0.66)

        if prediction >= high_thresh:
            st.success("High Yield Expected ✅")
        elif prediction >= low_thresh:
            st.warning("Moderate Yield Expected ⚖")
        else:
            st.error("Low Yield Expected ⚠")

# ============================
# OVERALL HISTORICAL TREND (ALL CROPS)
# ============================
st.write("---")
st.subheader("📊 Historical Yield Trend (All Crops & Seasons)")

selected_year_range = st.slider(
    "Select Year Range",
    min_value=2000,
    max_value=2015,
    value=(2000, 2015)
)

filtered_df = df_original[
    (df_original["Crop_Year"] >= selected_year_range[0]) &
    (df_original["Crop_Year"] <= selected_year_range[1])
]

yearly_trend = filtered_df.groupby("Crop_Year")["Yield"].mean().reset_index()

if not yearly_trend.empty:
    fig_trend = px.line(
        yearly_trend,
        x="Crop_Year",
        y="Yield",
        markers=True,
        title="Average Yield Across India (2000–2015)",
        template="plotly_dark"
    )
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.warning("No data available for selected range.")


# ----------------------------
# FOOTER
# ----------------------------
st.write("---")
st.caption("Advanced Crop Yield Prediction System | Overall & Crop-Specific Insights") 