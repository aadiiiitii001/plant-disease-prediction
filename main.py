import os
import json
import gdown
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# -----------------------------
# Paths and Model Setup
# -----------------------------
working_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(working_dir, "trained_model")
model_path = os.path.join(model_dir, "plant_disease_prediction_model.h5")

# Google Drive file link (Direct download ID)
drive_file_id = "1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf"
gdrive_url = f"https://drive.google.com/uc?id={drive_file_id}"

# Create directory if missing
os.makedirs(model_dir, exist_ok=True)

# -----------------------------
# Auto-download model if missing
# -----------------------------
if not os.path.exists(model_path):
    st.warning("Model file not found — downloading from Google Drive...")
    gdown.download(gdrive_url, model_path, quiet=False)
    st.success("✅ Model downloaded successfully!")

# -----------------------------
# Load Model and Classes
# -----------------------------
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Load class indices (make sure file exists)
class_indices_path = os.path.join(working_dir, "app", "class_indices.json")
if not os.path.exists(class_indices_path):
    # fallback if not in app folder
    class_indices_path = os.path.join(working_dir, "class_indices.json")

try:
    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)
except Exception as e:
    st.error(f"❌ Failed to load class indices: {e}")
    st.stop()

# -----------------------------
# Helper Functions
# -----------------------------
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess image for prediction."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.0
    return img_array


def predict_image_class(model, image, class_indices):
    """Predict image class and return the label."""
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌿 Plant Disease Classifier")
st.write("Upload a plant leaf image to predict the disease using a CNN model.")

uploaded_image = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image.resize((200, 200)), caption="Uploaded Image")

    with col2:
        if st.button("🔍 Classify"):
            try:
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f"🌱 Prediction: **{prediction}**")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
