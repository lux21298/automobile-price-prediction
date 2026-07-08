from __future__ import annotations

import pickle
import pandas as pd
import streamlit as st

from automobile_price.config import MODEL_PATH
from automobile_price.data import load_raw_data, prepare_data


st.set_page_config(page_title="Automobile Price Prediction")

st.title("Automobile Price Prediction")
st.write(
    "Estimate vehicle prices using a regression model trained on the UCI Automobile dataset."
)


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    return prepare_data(load_raw_data())


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


data = load_reference_data()
model = load_model()

with st.expander("Preview dataset", expanded=False):
    st.caption("Source: UCI Automobile dataset")
    st.dataframe(data.head(20), use_container_width=True)

if model is None:
    st.warning("Model artifact not found. Run `python -m automobile_price.train` first.")
    st.stop()

st.subheader("Vehicle Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    make = st.selectbox("Make", sorted(data["make"].dropna().unique()))
    fuel_type = st.selectbox("Fuel type", sorted(data["fuel-type"].dropna().unique()))
    body_style = st.selectbox("Body style", sorted(data["body-style"].dropna().unique()))
    drive_wheels = st.selectbox("Drive wheels", sorted(data["drive-wheels"].dropna().unique()))

with col2:
    engine_size = st.number_input("Engine size", min_value=50, max_value=400, value=120)
    horsepower = st.number_input("Horsepower", min_value=40, max_value=300, value=110)
    curb_weight = st.number_input("Curb weight", min_value=1400, max_value=4500, value=2500)
    highway_mpg = st.number_input("Highway MPG", min_value=10, max_value=60, value=30)

with col3:
    wheel_base = st.number_input("Wheel base", min_value=80.0, max_value=125.0, value=96.0)
    length = st.number_input("Length", min_value=140.0, max_value=210.0, value=175.0)
    width = st.number_input("Width", min_value=55.0, max_value=75.0, value=65.0)
    height = st.number_input("Height", min_value=45.0, max_value=65.0, value=53.0)

baseline = data.drop(columns=["price"]).median(numeric_only=True).to_dict()
sample = data.drop(columns=["price"]).mode(dropna=True).iloc[0].to_dict()
sample.update(baseline)
sample.update(
    {
        "make": make,
        "fuel-type": fuel_type,
        "body-style": body_style,
        "drive-wheels": drive_wheels,
        "engine-size": engine_size,
        "horsepower": horsepower,
        "curb-weight": curb_weight,
        "highway-mpg": highway_mpg,
        "wheel-base": wheel_base,
        "length": length,
        "width": width,
        "height": height,
    }
)

input_df = pd.DataFrame([sample])
prediction = model.predict(input_df)[0]

st.metric("Estimated price", f"${prediction:,.0f}")
st.caption("Prediction is an educational estimate based on a small historical dataset.")
