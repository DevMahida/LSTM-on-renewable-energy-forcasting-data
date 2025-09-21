import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load LSTM model
@st.cache_resource
def load_lstm_model():
    return load_model("renewable_energy_lstm_model.h5")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel("renewable_energy_forecasting_dataset.xlsx")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

# Title
st.title("⚡ Renewable Energy Forecasting with LSTM")

# Sidebar sliders
lookback = st.sidebar.slider("Lookback hours", 12, 72, 24)
forecast_horizon = st.sidebar.slider("Forecast horizon (hours)", 12, 72, 48)

# Load data & model
df = load_data()
st.write("### Sample Data", df.head())
model = load_lstm_model()

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Prepare input for forecasting
last_data = scaled_data[-lookback:]
current_batch = last_data.reshape((1, lookback, scaled_data.shape[1]))

future_forecast = []
for _ in range(forecast_horizon):
    next_pred = model.predict(current_batch, verbose=0)
    future_forecast.append(next_pred[0])
    
    next_features = current_batch[0, -1, :].copy()
    next_features[0:2] = next_pred[0]  # update solar & wind
    current_batch = np.append(current_batch[:, 1:, :], [[next_features]], axis=1)

# Inverse transform predictions
dummy_array = np.zeros((len(future_forecast), scaled_data.shape[1]))
dummy_array[:, 0:2] = np.array(future_forecast)
forecast_inversed = scaler.inverse_transform(dummy_array)[:, 0:2]

# Future dates
future_dates = pd.date_range(df.index[-1] + pd.Timedelta(hours=1), periods=forecast_horizon, freq="h")

# Plot
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(future_dates, forecast_inversed[:,0], label="Solar Forecast (kWh)", color="red")
ax.plot(future_dates, forecast_inversed[:,1], label="Wind Forecast (kWh)", color="blue")
ax.set_title("LSTM Renewable Energy Forecast")
ax.set_xlabel("Time")
ax.set_ylabel("kWh")
ax.legend()
st.pyplot(fig)
