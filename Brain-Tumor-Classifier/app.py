# app.py (Upgraded with Confidence Scores, Info, and Batch Upload)

import streamlit as st
from PIL import Image
import torch
from transformers import SegformerForImageClassification, SegformerImageProcessor
import os
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide" # Use wide layout for a better look
)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Load the fine-tuned model and processor from the nested directory."""
    model_path = "Brain-Tumor-Classifier/model"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = SegformerForImageClassification.from_pretrained(model_path).to(device)
        processor = SegformerImageProcessor.from_pretrained(model_path)
        return model, processor, device
    except Exception as e:
        return None, None, e

# --- NEW FEATURE: Tumor Information Dictionary ---
TUMOR_INFO = {
    "glioma": "A glioma is a common type of tumor originating in the glial cells that surround and support neurons in the brain. It is considered an aggressive, malignant (cancerous) tumor.",
    "meningioma": "A meningioma is a tumor that arises from the meninges — the membranes that surround the brain and spinal cord. Most meningiomas are noncancerous (benign).",
    "pituitary": "A pituitary tumor is an abnormal growth in the pituitary gland, a small gland at the base of the brain. Most of these tumors are benign and don't spread to other parts of the body.",
    "notumor": "This scan appears to be healthy, with no tumor detected by the model."
}

# --- Main Application ---
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("Upload one or more MRI scans. The AI will analyze each image and provide a classification, confidence score, and information about the tumor type.")

model, processor, device = load_model()

if model is None:
    st.error(f"Error loading the model: {device}")
    st.error("Please ensure the model files exist at the correct path within the repository.")
else:
    # --- NEW FEATURE: Allow multiple file uploads ---
    uploaded_files = st.file_uploader(
        "Choose MRI images...", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        # Create a list to store results for the table
        results_list = []

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            
            # --- AI Prediction Logic ---
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(device)
                logits = model(**inputs).logits
                
                # --- NEW FEATURE: Calculate probabilities ---
                probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
                top_prob = probabilities.max().item()
                predicted_label_id = probabilities.argmax().item()
                predicted_label = model.config.id2label[predicted_label_id]

            # Store result for the table
            results_list.append({
                "Filename": uploaded_file.name,
                "Prediction": predicted_label.capitalize(),
                "Confidence": f"{top_prob:.2%}"
            })

            # --- Display Individual Image Results ---
            st.subheader(f"Analysis for: {uploaded_file.name}")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption='Uploaded MRI Scan', use_container_width=True)
            with col2:
                # Display the main prediction
                if predicted_label == "notumor":
                    st.success(f"**Result:** The model predicts **No Tumor**.")
                else:
                    st.warning(f"**Result:** The model predicts a **{predicted_label.capitalize()}**.")
                
                # --- NEW FEATURE: Display the confidence score ---
                st.metric(label="Confidence Score", value=f"{top_prob:.2%}")
                
                # --- NEW FEATURE: Display the full probability chart ---
                st.write("Full Prediction Probabilities:")
                prob_df = pd.DataFrame(probabilities.cpu().numpy(), index=model.config.id2label.values(), columns=['Probability'])
                prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                st.bar_chart(probabilities.cpu().numpy())

            # --- NEW FEATURE: Display the tumor information ---
            with st.expander("Learn more about the prediction"):
                st.info(TUMOR_INFO[predicted_label])
            
            st.markdown("---")
            
        # --- NEW FEATURE: Display the summary table for batch uploads ---
        if len(uploaded_files) > 1:
            st.subheader("Batch Analysis Summary")
            summary_df = pd.DataFrame(results_list)
            st.dataframe(summary_df)

st.markdown("---")
st.markdown("Developed by Tiong. [View on GitHub](https://github.com/changvui/Brain-Tumor-Classifier.git)")
