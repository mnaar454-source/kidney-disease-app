import io
import json
import os
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import keras

from keras.applications.inception_v3 import preprocess_input
from PIL import Image, ImageOps


# =========================================================
# 1. CONFIGURATION
# =========================================================

MODEL_PATH = "kidney_inceptionv3.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (160, 160)  # Preserve the size used during training
MAX_FILE_MB = 10

DISCLAIMER = (
    "This tool is for educational/screening support only "
    "and is not a medical diagnosis."
)

st.set_page_config(
    page_title="KidneyAI | Smart CT Workspace",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

CONDITION_INFO = {
    "Normal": {
        "description": "The image was assigned to the Normal dataset category.",
        "details": "This result does not rule out a medical condition.",
        "next_step": "Discuss symptoms and scan findings with a qualified clinician."
    },
    "Cyst": {
        "description": "A kidney cyst is a fluid-filled sac. Many kidney cysts are benign.",
        "details": "Clinical significance depends on the cyst's appearance and other findings.",
        "next_step": "A clinician can determine whether further assessment is needed."
    },
    "Tumor": {
        "description": "A tumor is an abnormal growth of tissue.",
        "details": "This model cannot determine whether a growth is benign or malignant.",
        "next_step": "Seek professional review of the scan."
    },
    "Stone": {
        "description": "A kidney stone is a hard deposit of minerals and salts.",
        "details": "Symptoms may include side pain, blood in urine, nausea, or vomiting.",
        "next_step": "Seek medical advice. Severe pain with fever or inability to urinate needs urgent care."
    }
}


# =========================================================
# 2. SESSION STATE
# =========================================================

defaults = {
    "history": [],
    "results": {},
    "current_result": None,
    "dark_mode": False,
    "show_details": True,
    "case_counter": 0
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 3. INTERACTIVE VISUAL THEME
# =========================================================

with st.sidebar:
    st.markdown("# 🩺 KidneyAI")
    st.caption("Intelligent CT Image Workspace")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔬 Image Analysis",
            "⚖️ Compare Scans",
            "📚 Prediction History",
            "📈 Model Information",
            "ℹ️ About & Help"
        ]
    )

    st.divider()
    st.session_state.dark_mode = st.toggle(
        "🌙 Dark mode",
        value=st.session_state.dark_mode
    )

    st.session_state.show_details = st.toggle(
        "Show detailed explanations",
        value=st.session_state.show_details
    )

    st.divider()
    st.caption("B.Tech CSE (AI & ML)")
    st.caption("Quantum University, Roorkee")


if st.session_state.dark_mode:
    bg = "#101923"
    card = "#1b2937"
    text = "#edf5fb"
    muted = "#b3c3d0"
    border = "#344657"
else:
    bg = "#f2f7fb"
    card = "#ffffff"
    text = "#18364b"
    muted = "#687f90"
    border = "#dce8ef"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {bg};
        color: {text};
    }}
    [data-testid="stSidebar"] {{
        background: {card};
        border-right: 1px solid {border};
    }}
    .hero {{
        background: linear-gradient(120deg,#087f8c,#1766a5);
        padding: 30px;
        border-radius: 20px;
        color: white;
        margin-bottom: 24px;
    }}
    .hero h1 {{color:white;font-size:34px;}}
    .hero p {{color:#eaf8ff;line-height:1.8;}}
    .card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 12px;
    }}
    .card h3 {{color:#078b96;font-size:18px;}}
    .muted {{color:{muted};}}
    .disclaimer {{
        background:#fff6d9;
        color:#765900;
        padding:15px;
        border-left:5px solid #e7b52c;
        border-radius:10px;
        line-height:1.7;
        margin-top:20px;
    }}
    div.stButton > button, div.stDownloadButton > button {{
        border-radius:10px;
        min-height:42px;
        font-weight:600;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 4. LOAD MODEL ONLY ONCE
# =========================================================

@st.cache_resource
def load_model_and_classes():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Cannot find {MODEL_PATH}. Put the model beside app.py."
        )

    if not os.path.isfile(CLASS_NAMES_PATH):
        raise FileNotFoundError(
            f"Cannot find {CLASS_NAMES_PATH}."
        )

    model = keras.models.load_model(MODEL_PATH, compile=False)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        names = json.load(f)

    if isinstance(names, dict):
        names = [
            name for name, index in
            sorted(names.items(), key=lambda item: item[1])
        ]

    if len(names) != 4:
        raise ValueError(
            "Expected four class names. Check class_names.json."
        )

    return model, list(names)


# =========================================================
# 5. IMAGE UPLOAD AND PREPROCESSING
# =========================================================

def read_image(upload):
    if upload.size > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(f"Maximum image size is {MAX_FILE_MB} MB.")

    try:
        upload.seek(0)
        image = Image.open(upload)
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
        upload.seek(0)
        return image
    except Exception as exc:
        raise ValueError("This file is not a readable image.") from exc


def preprocess_image(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32)
    batch = preprocess_input(np.expand_dims(array, axis=0))
    return batch


# =========================================================
# 6. PREDICTION AND CONFIDENCE
# =========================================================

def analyse_image(image):
    model, class_names = load_model_and_classes()

    raw = np.asarray(
        model.predict(preprocess_image(image), verbose=0)
    ).squeeze()

    if raw.ndim != 1 or len(raw) != len(class_names):
        raise ValueError("Unexpected model output shape.")

    if not np.all(np.isfinite(raw)):
        raise ValueError("The model returned invalid values.")

    # This app expects a softmax output.
    if np.any(raw < 0) or not np.isclose(raw.sum(), 1, atol=0.02):
        raise ValueError(
            "Output does not look like probabilities. Check the "
            "model's final activation and preprocessing."
        )

    order = np.argsort(raw)[::-1]
    best = int(order[0])
    second = int(order[1])

    confidence = float(raw[best])
    gap = confidence - float(raw[second])

    # Interface-only thresholds, not clinical thresholds.
    if confidence >= 0.80 and gap >= 0.20:
        level = "High"
    elif confidence >= 0.55 and gap >= 0.10:
        level = "Medium"
    else:
        level = "Low"

    case_id = "KAI-" + uuid.uuid4().hex[:8].upper()

    return {
        "case_id": case_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": class_names[best],
        "confidence": confidence,
        "level": level,
        "gap": gap,
        "classes": class_names,
        "probabilities": raw.tolist(),
        "image": image.copy()
    }


# =========================================================
# 7. RESULTS CHARTS
# =========================================================

def probability_chart(result):
    df = pd.DataFrame({
        "Condition": result["classes"],
        "Probability (%)": np.array(result["probabilities"]) * 100
    }).sort_values("Probability (%)", ascending=True)

    fig = px.bar(
        df,
        x="Probability (%)",
        y="Condition",
        orientation="h",
        text=df["Probability (%)"].map(lambda x: f"{x:.2f}%"),
        color="Probability (%)",
        color_continuous_scale=["#b9e8e2", "#087f8c", "#1557a6"],
        title="Probability distribution across all classes"
    )

    fig.update_layout(
        height=350,
        xaxis_range=[0, 100],
        coloraxis_showscale=False,
        showlegend=False,
        margin=dict(l=10, r=10, t=60, b=20)
    )
    return fig


def confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={"suffix": "%"},
        title={"text": "Top-class model score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#087f8c"},
            "steps": [
                {"range": [0, 55], "color": "#f7dede"},
                {"range": [55, 80], "color": "#fff0c2"},
                {"range": [80, 100], "color": "#d8f2e8"}
            ]
        }
    ))
    fig.update_layout(height=280, margin=dict(t=50, b=10))
    return fig


# =========================================================
# 8. REPORT DOWNLOAD
# =========================================================

def report_json(result):
    return json.dumps({
        "Case ID": result["case_id"],
        "Date": result["date"],
        "Predicted class": result["prediction"],
        "Confidence score (%)": round(result["confidence"] * 100, 2),
        "Confidence indicator": result["level"],
        "Probabilities (%)": {
            name: round(float(prob) * 100, 2)
            for name, prob in zip(
                result["classes"], result["probabilities"]
            )
        },
        "Disclaimer": DISCLAIMER
    }, indent=2).encode("utf-8")


def report_csv(history):
    if not history:
        return b""
    return pd.DataFrame(history).to_csv(index=False).encode("utf-8")


# =========================================================
# 9. REUSABLE RESULT VIEW
# =========================================================

def show_result(result, label="Analysis result"):
    st.markdown(f"## 📊 {label}")

    a, b, c = st.columns(3)

    a.metric("Predicted category", result["prediction"])
    b.metric("Top-class score", f"{result['confidence'] * 100:.2f}%")
    c.metric("Confidence indicator", result["level"])

    if result["level"] == "Low":
        st.warning(
            "Low confidence: the model is uncertain. Review all class "
            "scores and do not interpret the prediction as a diagnosis."
        )
    elif result["gap"] < 0.10:
        st.warning(
            "The top two class scores are close. The model may not "
            "clearly distinguish between these categories."
        )
    else:
        st.info(
            "This confidence score reflects model output, not clinical certainty."
        )

    left, right = st.columns([1.25, 1])

    with left:
        st.plotly_chart(
            probability_chart(result),
            use_container_width=True,
            key=f"prob_{result['case_id']}"
        )

    with right:
        st.plotly_chart(
            confidence_gauge(result["confidence"]),
            use_container_width=True,
            key=f"gauge_{result['case_id']}"
        )

    if st.session_state.show_details:
        info = CONDITION_INFO.get(
            result["prediction"],
            {"description": "No description available.",
             "details": "", "next_step": ""}
        )

        with st.expander("📖 Understand this prediction", expanded=True):
            st.markdown(f"**About this category:** {info['description']}")
            st.write(info["details"])
            st.markdown(f"**General next step:** {info['next_step']}")

    with st.expander("🔎 View exact class scores"):
        for name, probability in zip(
            result["classes"], result["probabilities"]
        ):
            st.progress(
                float(probability),
                text=f"{name}: {probability * 100:.2f}%"
            )

    st.download_button(
        "⬇️ Download result JSON",
        data=report_json(result),
        file_name=f"{result['case_id']}_report.json",
        mime="application/json",
        key=f"download_{result['case_id']}"
    )

    st.markdown(
        f'<div class="disclaimer">⚠️ {DISCLAIMER}</div>',
        unsafe_allow_html=True
    )


# =========================================================
# 10. DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>KidneyAI Smart Analysis Workspace</h1>
        <p>
        A student research application for kidney CT image classification.
        Explore model scores, compare images, and review past predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    total = len(st.session_state.history)
    high_count = sum(
        1 for row in st.session_state.history
        if row["Confidence level"] == "High"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Session analyses", total)
    c2.metric("High-score results", high_count)
    c3.metric("Model categories", 4)

    st.markdown("## ✨ What would you like to do?")

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="card">
            <h3>🔬 Analyse a scan</h3>
            <p>Upload a CT image and explore all four class scores.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open analysis", use_container_width=True):
            st.session_state.quick_action = "analysis"

    with b:
        st.markdown("""
        <div class="card">
            <h3>⚖️ Compare scans</h3>
            <p>Upload two images and compare their predictions side by side.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open comparison", use_container_width=True):
            st.session_state.quick_action = "compare"

    with c:
        st.markdown("""
        <div class="card">
            <h3>📚 Review history</h3>
            <p>Inspect previous results and export the session table.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open history", use_container_width=True):
            st.session_state.quick_action = "history"

    st.markdown("## 🧬 Dataset categories")

    cols = st.columns(4)
    for col, name in zip(cols, ["Normal", "Cyst", "Tumor", "Stone"]):
        with col:
            st.markdown(f"""
            <div class="card">
                <h3>{name}</h3>
                <p>{CONDITION_INFO[name]['description']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.caption(
        "Use only images you are authorised to process. Avoid uploading "
        "identifiable patient information."
    )

    st.markdown(
        f'<div class="disclaimer">⚠️ {DISCLAIMER}</div>',
        unsafe_allow_html=True
    )


# =========================================================
# 11. IMAGE ANALYSIS
# =========================================================

elif page == "🔬 Image Analysis":

    st.markdown("""
    <div class="hero">
        <h1>Interactive Image Analysis</h1>
        <p>Upload a supported image to view model predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag and drop a JPG or PNG image",
        type=["jpg", "jpeg", "png"],
        key="single_upload"
    )

    if uploaded is not None:
        try:
            image = read_image(uploaded)

            col1, col2 = st.columns([1.2, 1])

            with col1:
                st.image(image, caption="Original uploaded image",
                         use_container_width=True)

            with col2:
                st.markdown("### Image information")
                st.write(f"Width: {image.width}px")
                st.write(f"Height: {image.height}px")
                st.write(f"File size: {uploaded.size / 1024:.1f} KB")

                preview_width = st.slider(
                    "Preview width",
                    min_value=250,
                    max_value=800,
                    value=500,
                    step=50
                )
                st.image(image, width=preview_width)

            if st.button("🚀 Run AI Analysis", type="primary",
                         use_container_width=True):
                with st.spinner("Analysing image with InceptionV3..."):
                    result = analyse_image(image)

                st.session_state.current_result = result
                st.session_state.results[result["case_id"]] = result

                st.session_state.history.insert(0, {
                    "Case ID": result["case_id"],
                    "Date": result["date"],
                    "Prediction": result["prediction"],
                    "Confidence (%)": round(result["confidence"] * 100, 2),
                    "Confidence level": result["level"]
                })

                st.success("Analysis finished.")

        except Exception as exc:
            st.error(f"Could not process image: {exc}")

    if st.session_state.current_result is not None:
        st.divider()
        show_result(st.session_state.current_result)


# =========================================================
# 12. TWO-IMAGE COMPARISON
# =========================================================

elif page == "⚖️ Compare Scans":

    st.markdown("""
    <div class="hero">
        <h1>Compare Two Images</h1>
        <p>Run the model separately on two images and compare its outputs.</p>
    </div>
    """, unsafe_allow_html=True)

    left_upload, right_upload = st.columns(2)

    with left_upload:
        file_a = st.file_uploader(
            "Upload image A",
            type=["jpg", "jpeg", "png"],
            key="compare_a"
        )

    with right_upload:
        file_b = st.file_uploader(
            "Upload image B",
            type=["jpg", "jpeg", "png"],
            key="compare_b"
        )

    if file_a is not None and file_b is not None:
        try:
            image_a = read_image(file_a)
            image_b = read_image(file_b)

            a, b = st.columns(2)
            with a:
                st.image(image_a, caption="Image A", use_container_width=True)
            with b:
                st.image(image_b, caption="Image B", use_container_width=True)

            if st.button("⚖️ Compare predictions", type="primary",
                         use_container_width=True):
                with st.spinner("Analysing both images..."):
                    result_a = analyse_image(image_a)
                    result_b = analyse_image(image_b)

                st.session_state.results[result_a["case_id"]] = result_a
                st.session_state.results[result_b["case_id"]] = result_b

                st.session_state.history.insert(0, {
                    "Case ID": result_a["case_id"],
                    "Date": result_a["date"],
                    "Prediction": result_a["prediction"],
                    "Confidence (%)": round(result_a["confidence"] * 100, 2),
                    "Confidence level": result_a["level"]
                })
                st.session_state.history.insert(0, {
                    "Case ID": result_b["case_id"],
                    "Date": result_b["date"],
                    "Prediction": result_b["prediction"],
                    "Confidence (%)": round(result_b["confidence"] * 100, 2),
                    "Confidence level": result_b["level"]
                })

                st.session_state.comparison = [result_a, result_b]

            if "comparison" in st.session_state:
                result_a, result_b = st.session_state.comparison

                c1, c2 = st.columns(2)
                with c1:
                    show_result(result_a, "Image A result")
                with c2:
                    show_result(result_b, "Image B result")

                comparison_df = pd.DataFrame({
                    "Class": result_a["classes"],
                    "Image A (%)": np.array(result_a["probabilities"]) * 100,
                    "Image B (%)": np.array(result_b["probabilities"]) * 100
                })

                fig = px.bar(
                    comparison_df,
                    x="Class",
                    y=["Image A (%)", "Image B (%)"],
                    barmode="group",
                    title="Side-by-side class score comparison"
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as exc:
            st.error(f"Comparison failed: {exc}")

    else:
        st.info("Upload both images to enable comparison.")


# =========================================================
# 13. HISTORY
# =========================================================

elif page == "📚 Prediction History":

    st.markdown("""
    <div class="hero">
        <h1>Prediction History</h1>
        <p>Search, filter and export results from the current session.</p>
    </div>
    """, unsafe_allow_html=True)

    history = st.session_state.history

    if history:
        df = pd.DataFrame(history)

        query = st.text_input("🔎 Search by case ID or prediction")

        if query:
            mask = (
                df["Case ID"].astype(str).str.contains(query, case=False)
                | df["Prediction"].astype(str).str.contains(query, case=False)
            )
            df = df[mask]

        class_filter = st.multiselect(
            "Filter predicted classes",
            options=["Normal", "Cyst", "Tumor", "Stone"]
        )

        if class_filter:
            df = df[df["Prediction"].isin(class_filter)]

        st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download filtered history",
            data=report_csv(df.to_dict("records")),
            file_name="kidneyai_history.csv",
            mime="text/csv"
        )

        if st.button("Clear all history"):
            st.session_state.history = []
            st.session_state.results = {}
            st.session_state.current_result = None
            st.session_state.pop("comparison", None)
            st.rerun()
    else:
        st.info("No history yet. Run an image analysis first.")


# =========================================================
# 14. MODEL INFORMATION
# =========================================================

elif page == "📈 Model Information":

    st.markdown("""
    <div class="hero">
        <h1>Model Performance & Architecture</h1>
        <p>Review model details and load evaluation metrics from your files.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Architecture summary")
    st.write("**Architecture:** InceptionV3 transfer learning")
    st.write(f"**Configured input size:** {IMG_SIZE[0]} × {IMG_SIZE[1]}")
    st.write("**Framework:** Keras")
    st.write("**Task:** Four-class image classification")

    st.markdown("### Load saved evaluation metrics")

    metrics_file = st.file_uploader(
        "Upload metrics.json (optional)",
        type=["json"],
        key="metrics_upload"
    )

    if metrics_file:
        try:
            metrics = json.load(metrics_file)
            st.json(metrics)
        except Exception:
            st.error("Unable to read metrics.json.")

    history_file = st.file_uploader(
        "Upload history.json (optional)",
        type=["json"],
        key="history_upload"
    )

    if history_file:
        try:
            history_data = json.load(history_file)
            history_data = history_data.get("history", history_data)

            if "accuracy" in history_data:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history_data["accuracy"],
                    mode="lines",
                    name="Training accuracy"
                ))
                if "val_accuracy" in history_data:
                    fig.add_trace(go.Scatter(
                        y=history_data["val_accuracy"],
                        mode="lines",
                        name="Validation accuracy"
                    ))
                fig.update_layout(
                    title="Training vs validation accuracy",
                    xaxis_title="Epoch",
                    yaxis_title="Accuracy"
                )
                st.plotly_chart(fig, use_container_width=True)

            if "loss" in history_data:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history_data["loss"],
                    mode="lines",
                    name="Training loss"
                ))
                if "val_loss" in history_data:
                    fig.add_trace(go.Scatter(
                        y=history_data["val_loss"],
                        mode="lines",
                        name="Validation loss"
                    ))
                fig.update_layout(
                    title="Training vs validation loss",
                    xaxis_title="Epoch",
                    yaxis_title="Loss"
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as exc:
            st.error(f"Could not read training history: {exc}")

    st.caption(
        "Do not report accuracy or other performance metrics unless they "
        "come from actual evaluation on held-out data."
    )


# =========================================================
# 15. ABOUT AND HELP
# =========================================================

elif page == "ℹ️ About & Help":

    st.markdown("""
    <div class="hero">
        <h1>About KidneyAI</h1>
        <p>Kidney Disease Detection using InceptionV3</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("How to use the application", expanded=True):
        st.write("1. Select Image Analysis.")
        st.write("2. Upload a JPG or PNG image.")
        st.write("3. Click Run AI Analysis.")
        st.write("4. Review the prediction and all class probabilities.")
        st.write("5. Download the result or review Prediction History.")

    with st.expander("Dataset and model"):
        st.write("Dataset: CT Kidney Dataset with Normal, Cyst, Tumor and Stone.")
        st.write("Model: InceptionV3 transfer learning.")
        st.write(f"Input size configured in this app: {IMG_SIZE}.")

    with st.expander("Limitations"):
        st.write(
            "File checks cannot prove an image is a CT scan. Model confidence "
            "is not clinical certainty. This application has not been "
            "validated for clinical diagnosis."
        )

    st.markdown("### Student details")
    st.write("Student: Add your name")
    st.write("Roll number: Add your roll number")
    st.write("Programme: B.Tech CSE (AI & ML)")
    st.write("Institution: Quantum School of Technology, Quantum University, Roorkee")

    st.markdown(
        f'<div class="disclaimer">⚠️ {DISCLAIMER}</div>',
        unsafe_allow_html=True
    )
