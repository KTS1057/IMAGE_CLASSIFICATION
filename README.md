<div align="center">

# 🧠 AI Vision Classifier — Multi-Category Deep Learning Engine

[![Live Demo](https://img.shields.io/badge/Live_App-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://imageclassification-9dbztmit3lrgj9yj3npfxb.streamlit.app/)
[![Model](https://img.shields.io/badge/Model-MobileNetV2_%2B_Custom_CNN-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://imageclassification-9dbztmit3lrgj9yj3npfxb.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Status](https://img.shields.io/badge/Status-Active_&_Upgraded-success?style=for-the-badge)](#)

*An end-to-end, multi-class Computer Vision platform recognizing domestic and wild animals, fruits, vegetables, and everyday objects with real-time domain analytics, feature map explainability, and edge detection.*

[🌐 Launch Live Web Application](https://imageclassification-9dbztmit3lrgj9yj3npfxb.streamlit.app/)

---

</div>

## 📌 Executive Overview

Traditional image classifiers are often constrained to rigid binary classification (such as Cat vs. Dog). This upgraded **AI Vision Classifier** transforms single-purpose recognition into an enterprise-grade multi-category visual recognition platform.

By coupling a pre-trained **MobileNetV2 Transfer Learning backbone** with an optional **Dedicated Custom Binary CNN**, this system can accurately classify over **1,000 object categories** (including hundreds of animal species and fruits/produce) in milliseconds. It augments raw probability outputs with contextual domain metadata:
- **🐾 Animals:** Displays biological taxonomy, characteristic traits, and behavioural trivia.
- **🍏 Fruits & Produce:** Displays estimated nutritional density (calories, vitamins, and health benefits).
- **🔬 Feature Visualizer:** Exposes spatial edges and convolutional gradient maps.

---

## ✨ Key Features & Capabilities

- **🔀 Dual Vision Engines:**
  - **🌟 Multi-Category Vision Engine:** Powered by MobileNetV2 pre-trained on ImageNet. Accurately identifies wild/domestic animals, fruits, produce, vehicles, and everyday items.
  - **🐾 Custom Cat vs. Dog CNN Model:** Preserves custom trained binary weights (`CAT_DOG_MODEL.h5`) for dedicated feline/canine comparative studies.
- **📸 Dual Input Modes:**
  - **File Uploader:** Drag-and-drop support for JPG, PNG, JPEG, and WEBP formats.
  - **Live Webcam Capture:** Real-time on-device camera input for immediate snapshot inference.
- **📊 Top-5 Probability Ranking:**
  - Color-coded confidence progress indicators with a customizable threshold filter.
- **🔬 Visual Explainability (Edge & Texture Maps):**
  - Convolutional spatial feature representation showing high-frequency gradient boundaries and edges that early CNN filters detect.
- **⚡ High-Throughput / Low Latency:**
  - Real-time inference latency reporting (typically `< 25ms` per sample).
- **📥 Diagnostic Report Generation:**
  - One-click export of inference diagnostics, image resolutions, and probability metrics to formatted JSON.

---

## 🛠️ Architecture & Model Comparison

```
[ Input Image ] ---> [ Preprocessing & Resize ] ---> [ Neural Backbone ]
                                                               |
                     +-----------------------------------------+
                     |                                         |
          [ MobileNetV2 Transfer Model ]             [ Custom Deep CNN ]
          (1,000 Multi-Class Head)                   (Binary Sigmoid Head)
                     |                                         |
           [ Top-5 Decoded Classes ]                [ Cat vs Dog Distribution ]
                     |                                         |
     +---------------+---------------+                         |
     |                               |                         |
[ Animal Taxonomy ]      [ Nutritional Facts ]                 |
     |                               |                         |
     +---------------+---------------+-------------------------+
                     |
       [ Interactive Streamlit Dashboard ]
```

| Metric / Dimension | Multi-Category MobileNetV2 | Custom Binary CNN |
| :--- | :--- | :--- |
| **Model Type** | Inverted Residual Deep CNN | Sequential Convolutional Network |
| **Total Parameters** | ~3.5 Million | ~2.5 Million |
| **Output Classes** | **1,000 Categories** (Animals, Fruits, Objects) | **2 Categories** (Cat vs Dog) |
| **Resolution Input** | 224 × 224 px | 160 × 160 px |
| **Feature Extraction** | Depthwise Separable Convolutions | Conv2D + MaxPooling2D |
| **Primary Use Case** | Broad visual taxonomy & produce ID | Focused domestic animal classification |

---

## 📁 Repository Structure

```text
IMAGE_CLASSIFICATION/
│
├── app.py                      # Upgraded Streamlit web application
├── train_custom_dataset.py     # End-to-end training pipeline for custom datasets
├── CAT_DOG_MODEL.h5            # Custom trained CNN model weights
├── requirements.txt            # Lightweight, pinned dependencies
└── README.md                   # Comprehensive project documentation
```

---

## 🚀 Training on Custom Animal or Fruit Datasets

This project includes a turnkey training pipeline ([`train_custom_dataset.py`](train_custom_dataset.py)) to fine-tune the vision model on your own local datasets (such as Kaggle's *Fruits-360* or *Animals-10*).

### 1. Organize Your Dataset
Structure your dataset folder by class names:
```text
my_dataset/
├── apple/
│   ├── img01.jpg
│   └── img02.jpg
├── banana/
│   ├── img01.jpg
│   └── ...
├── lion/
└── tiger/
```

### 2. Run the Training Script
```bash
python train_custom_dataset.py \
  --data_dir ./my_dataset \
  --epochs 15 \
  --batch_size 32 \
  --output_model custom_vision_classifier.keras
```

The script automatically:
1. Applies real-time data augmentations (random flips, rotations, zoom, and contrast shifts).
2. Performs Phase 1 training on top classification dense layers with frozen feature extractors.
3. Performs Phase 2 fine-tuning on high-level convolutional layers with adaptive learning rate decay.
4. Generates `class_labels.json` and saves the best model checkpoint.

---

## 💻 Local Installation & Setup

### Prerequisites
- Python 3.9, 3.10, or 3.11

### 1. Clone the Repository
```bash
git clone https://github.com/KTS1057/IMAGE_CLASSIFICATION.git
cd IMAGE_CLASSIFICATION
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🌐 Supported Datasets & Benchmarks

- **ImageNet (ILSVRC):** Benchmark foundation for MobileNetV2 backbone.
- **Kaggle Fruits-360:** 131 fruit and vegetable categories.
- **Kaggle Animals-10:** 10 diverse animal classes (dog, cat, horse, spyder, butterfly, chicken, sheep, cow, squirrel, elephant).
- **Microsoft Kaggle Dogs vs. Cats:** 25,000 labeled images for binary CNN validation.

---

## 👨‍💻 Author

**Kratarth Srivastava**  
Final Year B.Tech — Artificial Intelligence & Data Science, PSIT Kanpur  
- 🔗 **GitHub:** [@KTS1057](https://github.com/KTS1057)  
- 💼 **LinkedIn:** [kratarth-srivastava](https://www.linkedin.com/in/kratarth-srivastava/)  
- 📧 **Email:** kratarthsrivastava16@gmail.com
