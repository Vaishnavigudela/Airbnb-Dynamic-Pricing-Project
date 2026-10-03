import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Airbnb Dynamic Pricing Engine",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("airbnb_price_model.pkl")
model_features = joblib.load("model_features.pkl")


# ============================================================
# LOAD DATASET
# ============================================================

data = pd.read_csv("../data/AB_NYC_2019.csv")

data = data[
    (data["price"] > 0) &
    (data["price"] <= 1000)
].copy()

data["reviews_per_month"] = data["reviews_per_month"].fillna(0)


# ============================================================
# TITLE
# ============================================================

st.title("🏠 Airbnb Dynamic Pricing Recommendation Engine")

st.write(
    "Analyze Airbnb listing characteristics and generate "
    "a recommended nightly price using a machine learning model."
)


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("Listing Details")

neighbourhood_group = st.sidebar.selectbox(
    "Neighbourhood Group",
    sorted(data["neighbourhood_group"].unique())
)

neighbourhood = st.sidebar.selectbox(
    "Neighbourhood",
    sorted(data["neighbourhood"].unique())
)

room_type = st.sidebar.selectbox(
    "Room Type",
    sorted(data["room_type"].unique())
)

minimum_nights = st.sidebar.slider(
    "Minimum Nights",
    min_value=1,
    max_value=30,
    value=3
)

number_of_reviews = st.sidebar.slider(
    "Number of Reviews",
    min_value=0,
    max_value=500,
    value=20
)

reviews_per_month = st.sidebar.slider(
    "Reviews per Month",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

calculated_host_listings_count = st.sidebar.slider(
    "Host Listings Count",
    min_value=1,
    max_value=50,
    value=1
)

availability_365 = st.sidebar.slider(
    "Availability (Days per Year)",
    min_value=0,
    max_value=365,
    value=200
)


# ============================================================
# CREATE INPUT DATA
# ============================================================

input_data = pd.DataFrame({
    "neighbourhood_group": [neighbourhood_group],
    "neighbourhood": [neighbourhood],
    "room_type": [room_type],
    "minimum_nights": [minimum_nights],
    "number_of_reviews": [number_of_reviews],
    "reviews_per_month": [reviews_per_month],
    "calculated_host_listings_count": [calculated_host_listings_count],
    "availability_365": [availability_365]
})


# ============================================================
# ONE-HOT ENCODING
# ============================================================

input_data = pd.get_dummies(
    input_data,
    columns=[
        "neighbourhood_group",
        "room_type",
        "neighbourhood"
    ],
    drop_first=True
)


# Make input columns match training columns exactly

input_data = input_data.reindex(
    columns=model_features,
    fill_value=0
)


# ============================================================
# PRICE PREDICTION
# ============================================================

if st.sidebar.button("💰 Recommend Price"):

    predicted_price = model.predict(input_data)[0]

    st.success(
        f"Recommended Nightly Price: ${predicted_price:.2f}"
    )


# ============================================================
# DASHBOARD METRICS
# ============================================================

st.header("📊 Airbnb Market Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Listings",
    f"{len(data):,}"
)

col2.metric(
    "Average Price",
    f"${data['price'].mean():.2f}"
)

col3.metric(
    "Median Price",
    f"${data['price'].median():.2f}"
)

col4.metric(
    "Maximum Price",
    f"${data['price'].max():,.0f}"
)


# ============================================================
# PRICE BY ROOM TYPE
# ============================================================

st.subheader("Average Price by Room Type")

room_prices = (
    data.groupby("room_type")["price"]
    .mean()
    .sort_values(ascending=False)
)

st.bar_chart(room_prices)


# ============================================================
# PRICE BY NEIGHBOURHOOD GROUP
# ============================================================

st.subheader("Average Price by Neighbourhood Group")

group_prices = (
    data.groupby("neighbourhood_group")["price"]
    .mean()
    .sort_values(ascending=False)
)

st.bar_chart(group_prices)


# ============================================================
# TOP EXPENSIVE NEIGHBOURHOODS
# ============================================================

st.subheader("Top 10 Neighbourhoods by Average Price")

neighbourhood_prices = (
    data.groupby("neighbourhood")["price"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(neighbourhood_prices)

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("🤖 Pricing Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", "$50.10")
col2.metric("RMSE", "$89.37")
col3.metric("R² Score", "0.43")

st.info(
    "The Random Forest model explains approximately 41% of the "
    "variation in listing prices. The model is intended as a "
    "pricing recommendation tool rather than an exact price predictor."
)