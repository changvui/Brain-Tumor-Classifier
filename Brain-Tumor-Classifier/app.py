# app.py (Quick Fix Version)

import streamlit as st
from PIL import Image
import torch
from transformers import SegformerForImageClassification, SegformerImageProcessor
import os

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Load the fine-tuned model and processor from the nested directory."""
    # THE QUICK FIX IS HERE:
    model_path = "Brain-Tumor-Classifier/model"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = SegformerForImageClassification.from_pretrained(model_path).to(device)
        processor = SegformerImageProcessor.from_pretrained(model_path)
        return model, processor, device
    except Exception as e:
        return None, None, e

# --- Main Application ---
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("Upload an MRI scan of a brain. The AI will predict if it detects a glioma, meningioma, pituitary tumor, or no tumor.")

# (We can remove the debug line now)
# st.write("Current directory contents:", os.listdir("."))

model, processor, device = load_model()

if model is None:
    st.error(f"Error loading the model: {device}")
    st.error("Please ensure the model files exist at the correct path within the repository.")
else:
    st.success("AI model loaded successfully!")
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded MRI Scan', use_column_width=True)
        st.write("")

        if st.button('Analyze Image'):
            with st.spinner('The AI is thinking...'):
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits
                
                predicted_label_id = logits.argmax(-1).item()
                predicted_label = model.config.id2label[predicted_label_id]
                
                st.subheader("Analysis Complete!")
                if predicted_label == "notumor":
                    st.success(f"**Result:** The model predicts **No Tumor** was found.")
                else:
                    st.warning(f"**Result:** The model predicts a **{predicted_label.capitalize()}**.")

st.markdown("---")
st.markdown("Developed by Tiong. [View on GitHub](https://github.com/changvui/Brain-Tumor-Classifier.git)")
