import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import os

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data(file_path="renewable_energy_forecasting_dataset.xlsx"):
    if not os.path.exists(file_path):
        st.error(f"Dataset file not found: {file_path}")
        st.stop()
    df = pd.read_excel(file_path)
    return df

# -------------------------
# Rebuild LSTM Model & Load Weights
# -------------------------
@st.cache_resource
def load_lstm_model(weights_path="renewable_energy_lstm_model.h5"):
    """
    Rebuild the LSTM architecture manually and load saved weights.
    """
    if not os.path.exists(weights_path):
        st.error(f"Model file not found: {weights_path}")
        st.stop()

    n_timesteps = 24      # Last 24 steps for prediction
    n_features = 9        # Adjust to match dataset columns

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_timesteps, n_features)),
        Dense(n_features)
    ])
    model.load_weights(weights_path)
    return model

# -------------------------
# Main App
# -------------------------
st.title("Renewable Energy Forecasting using LSTM")

# Load data
df = load_data()
st.write("### Sample Data", df.head())

# Load model
model = load_lstm_model()

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)

# Predict next step
X_input = np.expand_dims(scaled_data[-24:], axis=0)  # (1, 24, n_features)
predicted_scaled = model.predict(X_input)
predicted = scaler.inverse_transform(predicted_scaled)

st.write("### Predicted Next Step", predicted)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(range(len(df)), df.values, label="Actual")
plt.plot(len(df), predicted[0], "ro", label="Predicted Next Step")
plt.legend()
st.pyplot(plt)
