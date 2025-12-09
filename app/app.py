import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model/stock_model.pkl", "rb"))


st.title("📈 Stock Market Price Prediction App")

st.write("Enter today's stock values to predict the NEXT closing price.")

open_p = st.number_input("Open Price", value=100.0)
high_p = st.number_input("High Price", value=101.0)
low_p = st.number_input("Low Price", value=98.0)
close_p = st.number_input("Close Price", value=99.0)
volume = st.number_input("Volume", value=100000)
market_cap = st.number_input("Market Cap", value=5e11)
pe_ratio = st.number_input("PE Ratio", value=20.0)

# Feature engineering (same as training)
ret = (close_p - open_p) / open_p
ma5 = close_p
ma10 = close_p

features = np.array([[open_p, high_p, low_p, close_p, volume,
                      market_cap, pe_ratio, ret, ma5, ma10]])

if st.button("Predict"):
    pred = model.predict(features)[0]
    st.success(f"Predicted Next Closing Price: {pred:.2f}")
