import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("E-Commerce Order Prediction System")

state = st.number_input("Customer State (encoded)", min_value=0)
qty = st.number_input("Quantity", min_value=1)
price = st.number_input("Product Price")
month = st.number_input("Order Month", min_value=1, max_value=12)

if st.button("Predict Order Status"):

    sample = pd.DataFrame({
        "Customer State":[state],
        "Quantity":[qty],
        "Price":[price],
        "Month":[month]
    })

    prediction = model.predict(sample)

    status_map = {
        0:"Cancelled",
        1:"Delivered",
        2:"RTO"
    }

    st.success("Prediction: " + status_map[prediction[0]])
