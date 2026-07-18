# CropGuard — AI-Powered Plant Disease Diagnosis

**Live demo:** https://cropguard-hg2m.onrender.com

*(Free-tier hosting — the first request may take 30-60s to wake the server. Subsequent requests are instant.)*

CropGuard identifies plant diseases from a single leaf photo. Upload an image, and it returns the most likely disease along with a confidence score — trained to recognize 38 crop-disease combinations across 14 species.

---

## How it works

Upload a photo of a leaf, the model analyzes it, and you get back a diagnosis with confidence scores, plus the next most likely alternatives.

- **Model:** MobileNetV2 (transfer learning)
- **Validation accuracy:** 99.10%
- **Classes:** 38 crop-disease combinations, 14 species
- **Dataset:** PlantVillage (~54,000 images) — https://www.kaggle.com/datasets/emmarex/plantdisease
- **Inference:** TensorFlow Lite (quantized, 2.5MB)
- **Backend:** Flask
- **Deployment:** Docker on Render

---

## Architecture and training

The model is a MobileNetV2 backbone pretrained on ImageNet, fine-tuned in two phases:

1. Phase 1, frozen backbone: only the classification head was trained, lr=1e-3, reaching 94.80% validation accuracy.
2. Phase 2, partial unfreeze: the top 30% of the backbone was unfrozen and fine-tuned at lr=1e-5, reaching 98.43%.
3. Phase 2B, gentle refinement: after a regression from an over-aggressive class-weighting experiment, the best checkpoint (98.98%) was restored and fine-tuned for 3 more epochs at lr=1e-6, reaching the final 99.10% validation accuracy.

Training used EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks, with class weighting to handle dataset imbalance across the 38 classes.

For deployment, the trained Keras model (best_model.h5, 24MB) was converted to a quantized TensorFlow Lite model (cropguard_quant.tflite, 2.5MB) — roughly a 90% size reduction with negligible accuracy loss, chosen for faster load times and a lighter deployment footprint.

---

## Project structure

    CropGuard/
    ├── app/
    │   ├── app.py                   Flask app, TFLite inference
    │   ├── static/
    │   │   ├── style.css
    │   │   └── main.js
    │   └── templates/
    │       └── index.html
    ├── models/
    │   ├── best_model.h5            Full Keras model, 99.10% val accuracy
    │   ├── cropguard_quant.tflite   Quantized model used in production
    │   └── class_names.json         Class index to disease label mapping
    ├── src/
    │   └── export.py                Keras to TFLite export script
    ├── Dockerfile
    ├── .dockerignore
    ├── requirements.txt
    └── README.md

---

## Running locally

Requirements: Python 3.12, pip

    git clone https://github.com/ananthan-bot/Cropguard.git
    cd Cropguard

    python3.12 -m venv venv
    source venv/bin/activate

    pip install -r requirements.txt

    python app/app.py

Then open http://localhost:5000 and upload a leaf photo.

---

## Running with Docker

    docker build -t cropguard .
    docker run -p 7860:7860 cropguard

Open http://localhost:7860.

---

## API

Besides the web UI, there is a JSON endpoint for programmatic predictions:

    curl -X POST -F "leaf_image=@path/to/leaf.jpg" https://cropguard-hg2m.onrender.com/predict

Example response:

    {
      "results": [
        {"label": "Strawberry - Leaf scorch", "confidence": 95.2},
        {"label": "Potato - Early blight", "confidence": 2.6},
        {"label": "Tomato - Early blight", "confidence": 0.8}
      ],
      "error": null
    }

---

## Tech stack

- Model training: TensorFlow / Keras, MobileNetV2, WSL2 Ubuntu with an NVIDIA RTX 4060 GPU
- Inference: TensorFlow Lite
- Backend: Flask
- Deployment: Docker, Render (free tier)
- Dataset: PlantVillage via Kaggle CLI

---

## Limitations

- Trained exclusively on the PlantVillage dataset, a controlled, mostly single-leaf, uniform-background dataset. Performance on real-world field photos with varied lighting, backgrounds, multiple leaves, or co-occurring diseases has not been separately validated and may be lower than the reported 99.10%.
- Covers 14 crop species; leaves from species outside this set will not be diagnosed correctly.
- Free-tier hosting means the server sleeps after 15 minutes of inactivity; the first request after a period of inactivity may take up to a minute to respond.

---

## License

MIT

---

## Acknowledgments

- PlantVillage dataset for the training data
- MobileNetV2 architecture (Sandler et al., 2018) via tf.keras.applications
