# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import load_model

# # Load your trained model (make sure you saved it earlier using model.save('model.h5'))
# @st.cache_resource
# def load_lstm_model():
#     return load_model("renewable_energy_lstm_model.h5")

# @st.cache_data
# def load_data():
#     # Replace with your actual dataset file
#     df_renewable_energy_forcasting = pd.read_excel('renewable_energy_forecasting_dataset.xlsx')
#     df_renewable_energy_forcasting['timestamp'] = pd.to_datetime(df_renewable_energy_forcasting['timestamp'])
#     df_renewable_energy_forcasting.set_index('timestamp', inplace=True)
#     return df_renewable_energy_forcasting

# st.title("⚡ Renewable Energy Forecasting with LSTM")

# # Sidebar inputs
# lookback = st.sidebar.slider("Lookback hours", min_value=12, max_value=72, value=24)
# forecast_horizon = st.sidebar.slider("Forecast horizon (hours)", min_value=12, max_value=72, value=48)

# df_renewable_energy_forcasting = load_data()
# st.write("### Sample Data", df_renewable_energy_forcasting.head())

# # Normalize
# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = scaler.fit_transform(df_renewable_energy_forcasting)

# model = load_lstm_model()

# # Prepare input for forecasting
# last_data = scaled_data[-lookback:]
# input_for_forecasting = last_data.reshape((1, lookback, scaled_data.shape[1]))

# future_forecast = []
# current_batch = input_for_forecasting.copy()

# for i in range(forecast_horizon):
#     next_pred = model.predict(current_batch, verbose=0)
#     future_forecast.append(next_pred[0])
    
#     next_features = current_batch[0, -1, :].copy()
#     next_features[0:2] = next_pred[0]  # update solar/wind only
    
#     current_batch = np.append(current_batch[:, 1:, :], [[next_features]], axis=1)

# dummy_array = np.zeros((len(future_forecast), scaled_data.shape[1]))
# dummy_array[:, 0:2] = np.array(future_forecast)
# forecast_inversed = scaler.inverse_transform(dummy_array)[:, 0:2]

# # Plot forecast
# future_dates = pd.date_range(df_renewable_energy_forcasting.index[-1] + pd.Timedelta(hours=1), periods=forecast_horizon, freq="h")
# fig, ax = plt.subplots(figsize=(12,6))
# ax.plot(future_dates, forecast_inversed[:,0], label="Solar Forecast (kWh)", color="red")
# ax.plot(future_dates, forecast_inversed[:,1], label="Wind Forecast (kWh)", color="blue")
# ax.set_title("LSTM Renewable Energy Forecast")
# ax.set_xlabel("Time")
# ax.set_ylabel("kWh")
# ax.legend()
# st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import onnxruntime as ort

# -------------------------
# Load ONNX LSTM Model
# -------------------------
@st.cache_resource
def load_lstm_model():
    return ort.InferenceSession("renewable_energy_lstm_model.onnx")

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("renewable_energy_forecasting_dataset.xlsx")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

# -------------------------
# Streamlit App
# -------------------------
st.title("⚡ Renewable Energy Forecasting (LSTM with ONNX)")

lookback = st.sidebar.slider("Lookback hours", 12, 72, 24)
forecast_horizon = st.sidebar.slider("Forecast horizon (hours)", 12, 72, 48)

df = load_data()
st.write("### Sample Data", df.head())

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Prepare input for forecasting
last_data = scaled_data[-lookback:]
input_batch = last_data.reshape(1, lookback, scaled_data.shape[1])

model = load_lstm_model()

# Forecasting loop
future_forecast = []
current_batch = input_batch.copy()

for _ in range(forecast_horizon):
    pred = model.run(None, {"input_1": current_batch.astype(np.float32)})[0]
    future_forecast.append(pred[0])

    # Update features for next timestep
    next_features = current_batch[0, -1, :].copy()
    next_features[0:2] = pred[0]  # update solar/wind outputs
    current_batch = np.append(current_batch[:, 1:, :], [[next_features]], axis=1)

# Inverse transform to original scale
dummy_array = np.zeros((len(future_forecast), scaled_data.shape[1]))
dummy_array[:, 0:2] = np.array(future_forecast)
forecast_inversed = scaler.inverse_transform(dummy_array)[:, 0:2]

# Create future dates
future_dates = pd.date_range(df.index[-1] + pd.Timedelta(hours=1), periods=forecast_horizon, freq="h")

# Plot forecast
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(future_dates, forecast_inversed[:,0], label="Solar Forecast (kWh)", color="red")
ax.plot(future_dates, forecast_inversed[:,1], label="Wind Forecast (kWh)", color="blue")
ax.set_title("LSTM Renewable Energy Forecast")
ax.set_xlabel("Time")
ax.set_ylabel("kWh")
ax.legend()
st.pyplot(fig)


