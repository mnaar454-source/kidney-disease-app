# Kidney Disease Detection using InceptionV3

B.Tech CSE (AI-ML) project. A Streamlit web app that classifies kidney CT images
into **Cyst, Normal, Stone, Tumor** using transfer learning with InceptionV3.

> **Disclaimer:** Academic research / decision-support prototype only.
> It is **not** a medical diagnostic tool. Consult a qualified medical professional.

## Files
- `app.py` – Streamlit web app
- `kidney_inceptionv3.keras` – trained model (Git LFS)
- `class_names.json` – class order used by the model
- `requirements.txt` – Python dependencies

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Live app
https://YOUR-APP-NAME.streamlit.app
