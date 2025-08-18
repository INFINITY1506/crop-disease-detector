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
    """Load model with maximum compatibility across TensorFlow/Keras versions."""
    
    # Try tf.keras first (most compatible with different versions)
    try:
        m = tf.keras.models.load_model(path, compile=False)
        st.info("✅ Model loaded successfully via tf.keras")
        return m
    except Exception as e1:
        st.warning(f"tf.keras loader failed: {str(e1)[:100]}...")
    
    # Try with custom objects if the first attempt failed
    try:
        # Define custom objects that might be missing
        custom_objects = {
            'Functional': tf.keras.Model,
        }
        m = tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects)
        st.info("✅ Model loaded with custom objects via tf.keras")
        return m
    except Exception as e2:
        st.warning(f"tf.keras with custom objects failed: {str(e2)[:100]}...")
    
    # Try pure Keras as last resort
    try:
        m = keras.models.load_model(path, safe_mode=False, compile=False)
        st.info("✅ Model loaded via Keras")
        return m
    except Exception as e3:
        st.error("❌ All model loading attempts failed")
        
        # Show a more helpful error message
        st.error("""
        **Model Loading Error**: The model file appears to be incompatible with the current TensorFlow/Keras version.
        
        **Possible solutions:**
        1. The model was saved with a different TensorFlow version
        2. Try re-saving the model with the current TensorFlow version
        3. Use TensorFlow SavedModel format instead of .keras format
        """)
        
        raise RuntimeError(
            f"Model loading failed with all methods. "
            f"tf.keras: {str(e1)[:50]}... | "
            f"custom_objects: {str(e2)[:50]}... | "
            f"keras: {str(e3)[:50]}..."
        )


@st.cache_resource
def load_model_and_maps():
    """Load model and supporting data with error handling."""
    try:
        # Try SavedModel format first (more compatible), then .keras format
        import os
        if os.path.exists("models/model_savedmodel"):
            st.info("🔄 Loading SavedModel format...")
            model = _load_any_model("models/model_savedmodel")
        else:
            st.info("🔄 Loading .keras format...")
            model = _load_any_model("models/model.keras")
        
        # Load class indices
        with open("models/class_indices.json") as f:
            idx2lbl = {int(k): v for k, v in json.load(f).items()}
        
        # Load disease information
        info = pd.read_csv("models/disease_info.csv")
        info_map = {row["label"]: row for _, row in info.iterrows()}
        
        st.success(f"✅ Successfully loaded model with {len(idx2lbl)} classes")
        return model, idx2lbl, info_map
        
    except Exception as e:
        st.error(f"❌ Failed to load model or data files: {str(e)}")
        st.info("""
        **Troubleshooting:**
        - Check that all required files are present in the models/ directory
        - If model loading fails, try running the convert_model.py script locally
        - The model might be incompatible with the current TensorFlow version
        """)
        raise

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
