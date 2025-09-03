# app.py (Upgraded with Grad-CAM Explainable AI)

import streamlit as st
from PIL import Image
import torch
from transformers import SegformerForImageClassification, SegformerImageProcessor
import os
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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
    model_path = "Brain-Tumor-Classifier/model"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = SegformerForImageClassification.from_pretrained(model_path).to(device)
        processor = SegformerImageProcessor.from_pretrained(model_path)
        return model, processor, device
    except Exception as e:
        return None, None, e

# --- GRAD-CAM FUNCTION ---
def generate_and_display_grad_cam(model, processor, original_image, predicted_label_id):
    """
    Generates and displays a Grad-CAM heatmap over the original image.
    """
    # 1. Preprocess the image for the model
    # Convert PIL Image to a NumPy array for OpenCV functions
    img_np = np.array(original_image)
    # Resize to the same size the model expects (usually 224x224 for ViT/SegFormer)
    img_resized = cv2.resize(img_np, (224, 224))
    # Normalize and create a tensor
    input_tensor = processor(images=img_resized, return_tensors="pt").pixel_values.to(device)

    # 2. Define the Target Layer in the SegFormer model
    # This is a bit tricky for Transformers. The last block of the encoder is a good choice.
    # For SegFormer-B0, this is a stable target.
    target_layers = [model.segformer.encoder.block[3][-1].layer_norm]
    
    # 3. Create the Grad-CAM object
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    
    # 4. Define the target for the CAM: which class are we explaining?
    targets = [ClassifierOutputTarget(predicted_label_id)]
    
    # 5. Generate the CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :] # Get the first (and only) CAM in the batch

    # 6. Overlay the CAM on the original image
    # We use a slightly transparent heatmap for better visualization
    visualization = show_cam_on_image(img_resized.astype(np.float32) / 255.0, grayscale_cam, use_rgb=True)

    # 7. Display the result
    st.subheader("AI's Focus Area (Heatmap)")
    st.image(visualization, caption="The red areas are what the AI focused on to make its prediction.", use_container_width=True)
    st.info("This visualization helps to understand which features in the MRI were most influential for the model's decision.")

# --- Main Application ---
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("Upload an MRI scan of a brain. The AI will predict if it detects a glioma, meningioma, pituitary tumor, or no tumor.")

model, processor, device = load_model()

if model is None:
    st.error(f"Error loading the model: {device}")
    st.error("Please ensure the model files exist at the correct path within the repository.")
else:
    st.success("AI model loaded successfully!")
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded MRI Scan', use_container_width=True)
        st.write("")

        if st.button('Analyze Image'):
            with st.spinner('The AI is thinking...'):
                # Process the image for prediction
                inputs = processor(images=image, return_tensors="pt").to(device)
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
                    
                    # --- NEW: Call the Grad-CAM function if a tumor is detected ---
                    try:
                        generate_and_display_grad_cam(model, processor, image, predicted_label_id)
                    except Exception as e:
                        st.error(f"Could not generate the heatmap. Error: {e}")

st.markdown("---")
st.markdown("Developed by Tiong. [View on GitHub](https://github.com/changvui/Brain-Tumor-Classifier.git)")
