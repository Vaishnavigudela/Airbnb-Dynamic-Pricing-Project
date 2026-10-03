import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("../data/AB_NYC_2019.csv")

print("Dataset loaded successfully!")
print("Shape:", data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())


# ============================================================
# 2. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")
print(data.isnull().sum())


# ============================================================
# 3. BASIC PRICE ANALYSIS
# ============================================================

print("\nPrice statistics:")
print(data["price"].describe())

print("\nRoom types:")
print(data["room_type"].value_counts())


# ============================================================
# 4. CLEAN PRICE DATA
# ============================================================

# Remove zero and extremely high prices
data = data[(data["price"] > 0) & (data["price"] <= 1000)]

print("\nAfter price cleaning:")
print("Rows remaining:", len(data))
print("Minimum price:", data["price"].min())
print("Maximum price:", data["price"].max())


# ============================================================
# 5. PRICE BY NEIGHBOURHOOD GROUP
# ============================================================

print("\nAverage price by neighbourhood group:")

print(
    data.groupby("neighbourhood_group")["price"]
    .mean()
    .sort_values(ascending=False)
)


# ============================================================
# 6. PRICE BY ROOM TYPE
# ============================================================

print("\nAverage price by room type:")

print(
    data.groupby("room_type")["price"]
    .mean()
    .sort_values(ascending=False)
)


# ============================================================
# 7. PRICE BY MINIMUM NIGHTS
# ============================================================

print("\nAverage price by minimum nights:")

print(
    data.groupby("minimum_nights")["price"]
    .mean()
    .sort_index()
    .head(20)
)


# ============================================================
# 8. PRICE BY NEIGHBOURHOOD
# ============================================================

print("\nTop 15 neighbourhoods by average price:")

print(
    data.groupby("neighbourhood")["price"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
)


# ============================================================
# 9. LISTING COUNT BY NEIGHBOURHOOD
# ============================================================

print("\nTop 15 neighbourhoods by number of listings:")

print(
    data["neighbourhood"]
    .value_counts()
    .head(15)
)


# ============================================================
# 10. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

model_data = data[
    [
        "neighbourhood_group",
        "neighbourhood",
        "room_type",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "price"
    ]
].copy()


# Missing review values mean there are no recorded reviews
model_data["reviews_per_month"] = (
    model_data["reviews_per_month"].fillna(0)
)


print("\nModel data:")
print(model_data.head())

print("\nModel data shape:", model_data.shape)


# ============================================================
# 11. SEPARATE FEATURES AND TARGET
# ============================================================

X = model_data.drop("price", axis=1)

y = model_data["price"]


# ============================================================
# 12. CONVERT CATEGORICAL DATA TO NUMBERS
# ============================================================

X = pd.get_dummies(
    X,
    columns=[
        "neighbourhood_group",
        "room_type",
        "neighbourhood"
    ],
    drop_first=True
)


# ============================================================
# 13. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ============================================================
# 14. TRAIN RANDOM FOREST MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=20,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nPricing model trained successfully!")


# ============================================================
# 15. MAKE PRICE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 16. EVALUATE MODEL
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\nModel Performance:")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 2))


# ============================================================
# 17. SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual Price": y_test.values[:10],
    "Predicted Price": np.round(y_pred[:10], 2)
})

print("\nSample Predictions:")
print(results)

import joblib

# Save model and feature columns
joblib.dump(model, "../dashboard/airbnb_price_model.pkl")
joblib.dump(X.columns.tolist(), "../dashboard/model_features.pkl")

print("\nPricing model saved successfully!")
