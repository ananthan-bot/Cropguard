"""
app/app.py

CropGuard demo: upload a leaf photo, get back the predicted disease + confidence.
Powered by the quantized TFLite model for lightweight deployment.

Usage:
    python app/app.py
    # then open http://localhost:5000
"""

import json
import os

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "..", "models", "cropguard_quant.tflite")
CLASS_NAMES_PATH = os.path.join(APP_DIR, "..", "models", "class_names.json")
IMG_SIZE = 224

app = Flask(__name__)

interpreter = None
input_details = None
output_details = None
class_names = None


def get_model():
    global interpreter, input_details, output_details, class_names
    if interpreter is None:
        print(f"Loading TFLite model from {MODEL_PATH} ...")
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)
    return interpreter, class_names


def prettify(label: str) -> str:
    if "___" in label:
        crop, disease = label.split("___", 1)
        return f"{crop.replace('_', ' ')} — {disease.replace('_', ' ')}"
    return label.replace("_", " ")


def predict_image(pil_image: Image.Image, top_k: int = 3):
    interp, names = get_model()

    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0).astype(input_details[0]["dtype"])

    interp.set_tensor(input_details[0]["index"], arr)
    interp.invoke()
    preds = interp.get_tensor(output_details[0]["index"])[0]

    top_indices = preds.argsort()[-top_k:][::-1]

    results = [
        {
            "label": prettify(names[str(i)]),
            "confidence": float(preds[i]) * 100,
        }
        for i in top_indices
    ]
    return results


@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    error = None

    if request.method == "POST":
        file = request.files.get("leaf_image")
        if not file or file.filename == "":
            error = "Please choose an image first."
        else:
            try:
                pil_image = Image.open(file.stream)
                results = predict_image(pil_image)
            except Exception as exc:
                error = f"Couldn't read that image: {exc}"

    return render_template("index.html", results=results, error=error)


@app.route("/predict", methods=["POST"])
def predict():
    """JSON endpoint for AJAX predictions — returns {results, error}."""
    file = request.files.get("leaf_image")
    if not file or file.filename == "":
        return jsonify({"error": "Please choose an image first.", "results": None}), 400
    try:
        pil_image = Image.open(file.stream)
        results = predict_image(pil_image)
        return jsonify({"results": results, "error": None})
    except Exception as exc:
        return jsonify({"error": f"Couldn't read that image: {exc}", "results": None}), 500


if __name__ == "__main__":
    get_model()
    app.run(debug=True, port=5000, host="0.0.0.0")
