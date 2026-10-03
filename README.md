# 🏠 Airbnb Dynamic Pricing Recommendation Engine

## 📌 Project Overview

The Airbnb Dynamic Pricing Recommendation Engine is a data analytics and machine learning project that analyzes Airbnb listing data and recommends a suitable nightly price based on listing characteristics.

The project combines exploratory data analysis, machine learning, and an interactive Streamlit dashboard to help understand Airbnb pricing patterns and generate data-driven price recommendations.

---

## 🎯 Objectives

- Analyze Airbnb listing prices across different locations and room types.
- Identify factors associated with Airbnb pricing.
- Clean and prepare the dataset for machine learning.
- Build a machine learning model to recommend listing prices.
- Create an interactive dashboard for users to experiment with listing characteristics.

---

## 📊 Dataset

The project uses the **New York City Airbnb Open Data (2019)** dataset.

The original dataset contains **48,895 Airbnb listings** and 16 attributes, including:

- Neighbourhood group
- Neighbourhood
- Room type
- Price
- Minimum nights
- Number of reviews
- Reviews per month
- Host listing count
- Availability

### Data Cleaning

Listings with a price of $0 and listings with prices above $1,000 were removed for the pricing analysis.

After cleaning, **48,645 listings** remained.

---

## 🔍 Exploratory Data Analysis

The analysis examined:

- Average price by neighbourhood group
- Average price by room type
- Average price by minimum nights
- Average price by neighbourhood
- Number of listings by neighbourhood

Some key observations include:

- Manhattan had the highest average price among the five neighbourhood groups.
- Entire home/apartment listings had a higher average price than private and shared rooms.
- Pricing varied considerably across neighbourhoods.

---

## 🤖 Machine Learning

A **Random Forest Regression** model was used to generate price recommendations.

### Features Used

- Neighbourhood group
- Neighbourhood
- Room type
- Minimum nights
- Number of reviews
- Reviews per month
- Calculated host listings count
- Availability

Categorical variables were converted into numerical features using one-hot encoding.

The dataset was divided into training and testing sets using an 80/20 split.

### Model Performance

| Metric | Result |
|---|---:|
| MAE | $50.87 |
| RMSE | $90.42 |
| R² Score | 0.41 |

The model explains approximately **41% of the variation in listing prices**. Therefore, it is intended as a **pricing recommendation tool rather than an exact price predictor**.

---

## 🖥️ Interactive Dashboard

A Streamlit dashboard was developed to allow users to enter listing characteristics and receive a recommended nightly price.

The dashboard includes:

- Neighbourhood selection
- Room type selection
- Minimum nights
- Number of reviews
- Reviews per month
- Host listings count
- Availability
- Recommended nightly price
- Airbnb market overview
- Average price by room type
- Average price by neighbourhood group
- Top neighbourhoods by average price
- Model performance metrics

### Example

For a selected listing configuration, the dashboard generates a machine-learning-based recommended nightly price.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Random Forest Regression**
- **Joblib**
- **Streamlit**

---

## 📁 Project Structure

```text
Airbnb_Pricing_Project/
│
├── README.md
│
├── data/
│   └── AB_NYC_2019.csv
│
├── python/
│   └── analysis.py
│
└── dashboard/
    ├── app.py
    ├── airbnb_price_model.pkl
    └── model_features.pkl