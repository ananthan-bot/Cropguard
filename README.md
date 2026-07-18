# 🌿 CropGuard — AI-Powered Plant Disease Diagnosis

![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-99.10%25-brightgreen)
![Model](https://img.shields.io/badge/Model-MobileNetV2-blue)
![Flask](https://img.shields.io/badge/Backend-Flask-black)
![TFLite](https://img.shields.io/badge/Inference-TensorFlow%20Lite-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Upload a leaf photo, get an instant disease diagnosis. Trained on 38 crop-disease combinations across 14 species, powered by a MobileNetV2 model fine-tuned to 99.10% validation accuracy.

---

## 🌐 Live Demo

| Service | URL |
|---|---|
| **Live app** | [cropguard-hg2m.onrender.com](https://cropguard-hg2m.onrender.com) |
| **API endpoint** | `POST /predict` |
| **Repository** | [github.com/ananthan-bot/Cropguard](https://github.com/ananthan-bot/Cropguard) |

*Free-tier hosting — the first request may take 30–60s to wake the server. Subsequent requests are instant.*

---

## ✨ Features

- 🔍 **Instant diagnosis** — upload a leaf photo, get a prediction in under a second
- 📊 **Confidence breakdown** — top prediction plus runner-up possibilities with confidence scores
- ⚡ **Lightweight inference** — quantized TensorFlow Lite model, 2.5MB
- 🖥️ **Clean web UI** — drag-and-drop upload, live results, no page reloads
- 🔌 **JSON API** — programmatic access via `/predict` for integrations
- 🐳 **Dockerized** — one-command reproducible deployment

---

## 📈 Model Performance

| Metric | Value |
|---|---|
| Architecture | MobileNetV2 (transfer learning) |
| Final validation accuracy | **99.10%** |
| Classes | 38 crop-disease combinations |
| Crop species covered | 14 |
| Dataset | PlantVillage (~54,000 images) |
| Deployed model size | 2.5MB (quantized TFLite) |
| Full model size | 24MB (Keras .h5) |

### Training progression

| Phase | Description | Val. Accuracy |
|---|---|---|
| 1 | Frozen backbone, lr=1e-3 | 94.80% |
| 2 | Top 30% unfrozen, lr=1e-5 | 98.43% |
| 2B | Restored best checkpoint, gentle fine-tune at lr=1e-6 | **99.10%** |

Training used `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` callbacks, with class weighting to handle imbalance across the 38 classes.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model training | TensorFlow / Keras, MobileNetV2 |
| Training hardware | WSL2 Ubuntu, NVIDIA RTX 4060 GPU |
| Inference | TensorFlow Lite |
| Backend | Flask |
| Deployment | Docker, Render (free tier) |
| Dataset | PlantVillage, via Kaggle CLI |

---

## 📁 Project Structure

```
CropGuard/
├── app/
│   ├── app.py                   # Flask app — TFLite inference
│   ├── static/
│   │   ├── style.css
│   │   └── main.js
│   └── templates/
│       └── index.html
├── models/
│   ├── best_model.h5            # Full Keras model, 99.10% val accuracy
│   ├── cropguard_quant.tflite   # Quantized model used in production
│   └── class_names.json         # Class index → disease label mapping
├── src/
│   └── export.py                # Keras → TFLite export script
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## 🚀 Running Locally

**Requirements:** Python 3.12, pip

```bash
git clone https://github.com/ananthan-bot/Cropguard.git
cd Cropguard

python3.12 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python app/app.py
```

Open **http://localhost:5000**.

---

## 🐳 Running with Docker

```bash
docker build -t cropguard .
docker run -p 7860:7860 cropguard
```

Open **http://localhost:7860**.

---

## 🔌 API Usage

```bash
curl -X POST -F "leaf_image=@path/to/leaf.jpg" https://cropguard-hg2m.onrender.com/predict
```

**Response:**

```json
{
  "results": [
    {"label": "Strawberry — Leaf scorch", "confidence": 95.2},
    {"label": "Potato — Early blight", "confidence": 2.6},
    {"label": "Tomato — Early blight", "confidence": 0.8}
  ],
  "error": null
}
```

---

## ⚠️ Limitations

- Trained exclusively on PlantVillage — a controlled, mostly single-leaf, uniform-background dataset. Real-world field photos (varied lighting, backgrounds, multiple leaves, co-occurring diseases) may see lower accuracy than the reported 99.10%.
- Covers 14 crop species only; other species are not supported.
- Free-tier hosting sleeps after 15 minutes of inactivity — first request after idle time may take up to a minute.

---

## 📄 License

MIT

---

## 🙏 Acknowledgments

- [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- MobileNetV2 (Sandler et al., 2018) via `tf.keras.applications`
