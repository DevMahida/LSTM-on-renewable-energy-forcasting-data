# convert_to_onnx.py
import keras2onnx
from tensorflow.keras.models import load_model
import onnx

# Load your trained Keras model
model = load_model("renewable_energy_lstm_model.h5")

# Convert to ONNX
onnx_model = keras2onnx.convert_keras(model, model.name)

# Save the ONNX model
onnx.save_model(onnx_model, "renewable_energy_lstm_model.onnx")

print("✅ Conversion complete! ONNX model saved as renewable_energy_lstm_model.onnx")
