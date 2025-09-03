# app.py (Final Professional Version with 4 Examples)

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
    layout="wide"
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

# --- TUMOR INFO DICTIONARY ---
TUMOR_INFO = {
    "glioma": "A glioma is a common type of tumor originating in the glial cells that surround and support neurons in the brain. It is considered an aggressive, malignant (cancerous) tumor.",
    "meningioma": "A meningioma is a tumor that arises from the meninges — the membranes that surround the brain and spinal cord. Most meningiomas are noncancerous (benign).",
    "pituitary": "A pituitary tumor is an abnormal growth in the pituitary gland, a small gland at the base of the brain. Most of these tumors are benign and don't spread to other parts of the body.",
    "notumor": "This scan appears to be healthy, with no tumor detected by the model."
}

# --- AI PREDICTION FUNCTION ---
def predict(image):
    """Takes a PIL image and returns the prediction and probabilities."""
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        logits = model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        top_prob = probabilities.max().item()
        predicted_label_id = probabilities.argmax().item()
        predicted_label = model.config.id2label[predicted_label_id]
    return predicted_label, top_prob, probabilities

# --- MAIN APP LAYOUT ---
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("This AI application analyzes brain MRI scans to classify them into one of four categories: **glioma, meningioma, pituitary tumor,** or **no tumor**.")

# Load the model
model, processor, device = load_model()

if model is None:
    st.error(f"Error loading the model: {device}")
else:
    # --- SIDEBAR FOR UPLOADS AND EXAMPLES ---
    st.sidebar.header("Upload or Select an Image")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload your MRI scan", 
        type=["jpg", "jpeg", "png"]
    )
    
    # Example images
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Or try an example:**")
    
    # Define example paths relative to the nested structure
    example_base_path = "Brain-Tumor-Classifier/examples/"
    
    if st.sidebar.button("Glioma Example"):
        uploaded_file = os.path.join(example_base_path, "glioma_example.jpg")
    if st.sidebar.button("Meningioma Example"):
        uploaded_file = os.path.join(example_base_path, "meningioma_example.jpg")
    
    # --- NEW: Added Pituitary Example Button ---
    if st.sidebar.button("Pituitary Example"):
        uploaded_file = os.path.join(example_base_path, "pituitary_example.jpg")
        
    if st.sidebar.button("No Tumor Example"):
        uploaded_file = os.path.join(example_base_path, "notumor_example.jpg")

    # --- ANALYSIS AND DISPLAY ---
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            
            # Perform prediction
            predicted_label, top_prob, probabilities = predict(image)
            
            # Main content area
            st.header("Analysis Result")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(image, caption='Uploaded MRI Scan', use_container_width=True)
            
            with col2:
                if predicted_label == "notumor":
                    st.success(f"**Result:** The model predicts **No Tumor**.")
                else:
                    st.warning(f"**Result:** The model predicts a **{predicted_label.capitalize()}**.")
                
                st.metric(label="Confidence Score", value=f"{top_prob:.2%}")
                
                st.write("Full Prediction Probabilities:")
                prob_df = pd.DataFrame(probabilities.cpu().numpy(), index=model.config.id2label.values(), columns=['Probability'])
                st.bar_chart(prob_df)

            with st.expander("Learn more about the prediction"):
                st.info(TUMOR_INFO[predicted_label])
            
            # --- User Feedback ---
            st.markdown("---")
            st.subheader("Was this prediction helpful?")
            feedback_col1, feedback_col2, _ = st.columns([1, 1, 4])
            if feedback_col1.button("👍 Yes"):
                st.success("Thank you for your feedback!")
            if feedback_col2.button("👎 No"):
                st.success("Thank you for your feedback! This helps us improve.")
        except FileNotFoundError:
            st.error("Example file not found. Please ensure 'pituitary_example.jpg' is in the 'examples' folder on GitHub.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- Project Details Section ---
st.markdown("---")
with st.expander("About This Project & Disclaimer"):
    st.markdown("""
        **Model:** `SegFormer-B0` (fine-tuned)  
        **Dataset:** Brain Tumor MRI Dataset from Kaggle  
        **Accuracy:** The model achieved **98% accuracy** on the test set.

        **Confusion Matrix:**
    """)
    # IMPORTANT: Make sure this image exists in your 'examples' folder on GitHub
    st.image(os.path.join(example_base_path, "confusion_matrix.png"))
    st.markdown("""
        ---
        **DISCLAIMER:** This AI application is a demonstration project and is **NOT a medical device**. 
        The predictions are not a substitute for professional medical advice, diagnosis, or treatment. 
        Always consult with a qualified healthcare provider for any health concerns.
    """)
