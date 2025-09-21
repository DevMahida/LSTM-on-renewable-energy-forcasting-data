import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data(file_path="renewable_energy_forecasting_dataset"):
    df = pd.read_excel(file_path)
    return df

# -------------------------
# Load LSTM Model
# -------------------------
@st.cache_resource
def load_lstm_model(model_path="renewable_energy_lstm_model"):
    """
    Load LSTM model saved in TensorFlow SavedModel format.
    Use compile=False to avoid InputLayer deserialization issues.
    """
    model = load_model(model_path, compile=False)
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

# Predict next step (example)
# Assume last 24 timesteps for prediction
X_input = np.expand_dims(scaled_data[-24:], axis=0)  # shape (1, 24, n_features)
predicted_scaled = model.predict(X_input)
predicted = scaler.inverse_transform(predicted_scaled)

st.write("### Predicted Next Step", predicted)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(range(len(df)), df.values, label="Actual")
plt.plot(len(df), predicted[0], "ro", label="Predicted Next Step")
plt.legend()
st.pyplot(plt)
