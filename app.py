import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import streamlit as st, pandas as pd, numpy as np
from PIL import Image
import tensorflow as tf
import keras
import json


st.set_page_config(page_title="Crop Disease Detector", page_icon="🌿", layout="centered")

def _sniff_model_format(path: str) -> str:
    """Best-effort detection of model format."""
    import os
    if os.path.isdir(path):
        return "savedmodel" if os.path.exists(os.path.join(path, "saved_model.pb")) else "dir"
    try:
        with open(path, "rb") as f:
            sig = f.read(8)
    except Exception:
        return "unknown"
    # Keras 3 .keras files are ZIPs (PK\x03\x04)
    if sig.startswith(b"PK\x03\x04"):
        return "keras_v3"
    # HDF5 starts with \x89HDF\r\n\x1a\n
    if sig.startswith(b"\x89HDF"):
        return "h5"
    return "unknown"

def _load_any_model(path: str):
    fmt = _sniff_model_format(path)
    # Prefer tf.keras for SavedModel dirs and H5
    if fmt in ("savedmodel", "dir"):
        try:
            m = tf.keras.models.load_model(path, compile=False)
            st.info("Loaded TensorFlow SavedModel/Directory via tf.keras.")
            return m
        except Exception as e:
            st.warning(f"tf.keras SavedModel loader failed: {e}")
    if fmt == "h5":
        try:
            m = tf.keras.models.load_model(path, compile=False)
            st.info("Loaded H5 via tf.keras.")
            return m
        except Exception as e:
            st.warning(f"tf.keras H5 loader failed: {e}")
    # Default & .keras path: try Keras 3 first, then tf.keras fallback
    try:
        m = keras.models.load_model(path, safe_mode=False, compile=False)
        st.info("Loaded .keras via Keras 3.")
        return m
    except Exception as e1:
        st.warning(f"Keras loader failed: {e1}. Trying tf.keras…")
        try:
            m = tf.keras.models.load_model(path, compile=False)
            st.info("Loaded via tf.keras fallback.")
            return m
        except Exception as e2:
            st.error("Couldn’t load model with either Keras or tf.keras.")
            raise RuntimeError(
                "Model load failed. First error (keras): " + str(e1) +
                " | Second error (tf.keras): " + str(e2)
            )


@st.cache_resource
def load_model_and_maps():
    # Robust model loader that handles .keras (Keras 3), H5, and SavedModel
    model = _load_any_model("models/model.keras")

    with open("models/class_indices.json") as f:
        idx2lbl = {int(k): v for k, v in json.load(f).items()}

    info = pd.read_csv("models/disease_info.csv")
    info_map = {row["label"]: row for _, row in info.iterrows()}
    return model, idx2lbl, info_map

model, idx2lbl, info_map = load_model_and_maps()
IMG_SIZE = (192, 192)

def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return x

def predict_pil(pil_img):
    x = preprocess(pil_img)
    preds = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    top_lbl = idx2lbl[top_idx]
    conf = float(preds[top_idx])
    top3_idx = preds.argsort()[-3:][::-1]
    top3 = [(idx2lbl[int(i)], float(preds[int(i)])) for i in top3_idx]
    return top_lbl, conf, top3

def show_report(lbl, conf):
    st.subheader(f"Prediction: {lbl} ({conf*100:.1f}%)")
    info = info_map.get(lbl)
    if info is not None:
        st.markdown(f"**Disease/Condition:** {info.get('title','')}")
        st.markdown(f"**Description:** {info.get('description','')}")
        st.markdown(f"**Treatment:** {info.get('treatment','')}")
        st.markdown(f"**Prevention:** {info.get('prevention','')}")
        ref = info.get('reference','')
        if isinstance(ref, str) and ref.strip():
            st.caption(f"Reference: {ref}")
    else:
        st.info("No info entry found for this class. Edit assets/disease_info.csv to add details.")

st.title("🌿 Crop Disease Detection")
st.write("Upload an image or use your camera. The app predicts the disease and shows treatment & prevention tips.")

tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Live Camera"])

with tab1:
    f = st.file_uploader("Choose a leaf image", type=["jpg","jpeg","png","webp"])
    if f is not None:
        img = Image.open(f)
        st.image(img, caption="Input", use_column_width=True)
        lbl, conf, top3 = predict_pil(img)
        show_report(lbl, conf)
        with st.expander("Top-3 predictions"):
            for l, c in top3:
                st.write(f"{l}: {c*100:.1f}%")

with tab2:
    cam = st.camera_input("Take a photo")
    if cam is not None:
        img = Image.open(cam)
        st.image(img, caption="Captured", use_column_width=True)
        lbl, conf, top3 = predict_pil(img)
        show_report(lbl, conf)
        with st.expander("Top-3 predictions"):
            for l, c in top3:
                st.write(f"{l}: {c*100:.1f}%")

st.caption("⚠️ Results are probabilistic. Confirm with local agronomy/extension services for critical decisions.")
