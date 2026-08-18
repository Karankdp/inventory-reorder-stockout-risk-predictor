import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Inventory Reorder & Stockout Risk", layout="centered")

@st.cache_resource
def load_artifact(_mtime):
    return joblib.load("inventory_model.joblib")

artifact = load_artifact(os.path.getmtime("inventory_model.joblib"))
model = artifact["model"]
classifier = artifact["classifier"]
scaler = artifact["scaler"]
feature_columns = artifact["feature_columns"]
cat_cols = artifact["cat_cols"]
resid_std = artifact["resid_std"]
Z = artifact["Z"]
threshold = artifact["deviation_threshold_pct"]
strongest_feature = artifact["strongest_feature"]

st.title("Inventory Reorder & Stockout Risk Checker")
st.write("Enter a SKU's details to get its predicted demand, reorder point, and risk flag.")

col1, col2 = st.columns(2)
with col1:
    category = st.selectbox("Category", ["Electronics", "Clothing", "Home & Garden", "Sports", "Toys"])
    region = st.selectbox("Region", ["North", "South", "East", "West"])
    weather = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy", "Snowy"])
    season = st.selectbox("Seasonality", ["Spring", "Summer", "Autumn", "Winter"])
    promo = st.selectbox("Holiday/Promotion", [0, 1])
with col2:
    inventory_level = st.number_input("Current Inventory Level", min_value=0, value=100)
    price = st.number_input("Price", min_value=0.0, value=50.0)
    discount = st.number_input("Discount (%)", min_value=0, max_value=100, value=0)
    competitor_price = st.number_input("Competitor Pricing", min_value=0.0, value=52.0)
    lead_time = st.number_input("Lead Time (days)", min_value=1, value=7)

units_ordered = st.number_input("Units Ordered (restock qty)", min_value=0, value=60)

if st.button("Check Reorder Risk"):
    row = {
        "Store ID": "S001", "Product ID": "P0001", "Category": category, "Region": region,
        "Inventory Level": inventory_level, "Units Ordered": units_ordered, "Price": price,
        "Discount": discount, "Weather Condition": weather, "Holiday/Promotion": promo,
        "Competitor Pricing": competitor_price, "Seasonality": season,
        "Month": 1, "DayOfWeek": 2, "IsWeekend": 0, "LeadTimeDays": lead_time,
    }
    row_df = pd.DataFrame([row])
    row_encoded = pd.get_dummies(row_df, columns=cat_cols, drop_first=True)
    row_encoded = row_encoded.reindex(columns=feature_columns, fill_value=0)
    row_scaled = scaler.transform(row_encoded)
    expected_n_features = getattr(model, "n_features_in_", row_scaled.shape[1])
    row_model_input = row_encoded[[strongest_feature]].values if expected_n_features == 1 else row_scaled

    predicted_demand = model.predict(row_model_input)[0]
    safety_stock = Z * resid_std * np.sqrt(lead_time)
    reorder_point = predicted_demand * lead_time + safety_stock
    deviation_pct = (inventory_level - reorder_point) / reorder_point * 100
    clf_proba = classifier.predict_proba(row_scaled)[0, 1]

    if deviation_pct <= -threshold:
        flag, color = "Understocked (High Risk)", "red"
    elif deviation_pct >= threshold:
        flag, color = "Overstocked", "orange"
    else:
        flag, color = "Optimal", "green"

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Predicted Daily Demand", f"{predicted_demand:.1f} units")
        st.metric("Reorder Point", f"{reorder_point:.1f} units")
        st.metric("Deviation from Reorder Point", f"{deviation_pct:+.1f}%")
    with col_b:
        st.metric("Classifier Stockout Probability", f"{clf_proba:.1%}")

    st.markdown(f"### Business-Rule Flag: :{color}[{flag}]")
    if clf_proba >= 0.5 and flag != "Understocked (High Risk)":
        st.warning("The direct classifier disagrees with the reorder-point rule — worth a manual look.")
