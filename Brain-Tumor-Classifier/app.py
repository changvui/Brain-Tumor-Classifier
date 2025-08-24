# app.py

import streamlit as st
from PIL import Image
import torch
from transformers import SegformerForImageClassification, SegformerImageProcessor

# Set page configuration for a professional look
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)

# --- MODEL LOADING ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    """Load the fine-tuned model and processor from the local 'model' directory."""
    model_path = "./model"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = SegformerForImageClassification.from_pretrained(model_path).to(device)
        processor = SegformerImageProcessor.from_pretrained(model_path)
        return model, processor, device
    except Exception as e:
        # If loading fails, return None and the error
        return None, None, e

# --- Main Application ---
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("Upload an MRI scan of a brain. The AI will predict if it detects a glioma, meningioma, pituitary tumor, or no tumor.")

# Load the model and handle potential errors
model, processor, device = load_model()

if model is None:
    st.error(f"Error loading the model: {device}")
    st.error("Please ensure the 'model' directory is in the same folder as this app and contains the correct files.")
else:
    st.success("AI model loaded successfully!")

    # --- Image Uploader ---
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded MRI Scan', use_column_width=True)
        st.write("") # Add a little space

        # Add a button to trigger the classification
        if st.button('Analyze Image'):
            with st.spinner('The AI is thinking...'):
                # Process the image
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                # Make a prediction
                with torch.no_grad():
                    logits = model(**inputs).logits
                
                predicted_label_id = logits.argmax(-1).item()
                predicted_label = model.config.id2label[predicted_label_id]
                
                # --- Display the Result ---
                st.subheader("Analysis Complete!")
                if predicted_label == "notumor":
                    st.success(f"**Result:** The model predicts **No Tumor** was found.")
                else:
                    st.warning(f"**Result:** The model predicts a **{predicted_label.capitalize()}**.")

st.markdown("---")
st.markdown("Developed by [Your Name]. [View on GitHub](https://github.com/changvui/Brain-Tumor-Classifier.git)")