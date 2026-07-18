"""
src/export.py

Converts the trained Keras model to TFLite (float32 + quantized)
for lightweight deployment.
"""

import tensorflow as tf
import os
import numpy as np

MODEL_PATH = "best_model.h5"
OUT_DIR = "models"

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading Keras model from {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Input shape:", model.input_shape)

# --- Float32 TFLite ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
float_path = os.path.join(OUT_DIR, "cropguard.tflite")
with open(float_path, "wb") as f:
    f.write(tflite_model)
print(f"Saved {float_path} ({os.path.getsize(float_path)/1e6:.2f} MB)")

# --- Quantized TFLite (dynamic range) ---
converter_q = tf.lite.TFLiteConverter.from_keras_model(model)
converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter_q.convert()
quant_path = os.path.join(OUT_DIR, "cropguard_quant.tflite")
with open(quant_path, "wb") as f:
    f.write(tflite_quant_model)
print(f"Saved {quant_path} ({os.path.getsize(quant_path)/1e6:.2f} MB)")

# --- Sanity check ---
interpreter = tf.lite.Interpreter(model_path=quant_path)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
dummy = np.random.rand(*inp["shape"]).astype(inp["dtype"])
interpreter.set_tensor(inp["index"], dummy)
interpreter.invoke()
result = interpreter.get_tensor(out["index"])
print("Sanity check passed. Output shape:", result.shape)
