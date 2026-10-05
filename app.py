import streamlit as st
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

# --- 1. Load Model (cached so it only loads once) ---
@st.cache_resource
def load_model():
    model_path = "YOUR_USERNAME/knee-oa-vit-classifier"  # <-- Replace with your HF username
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTForImageClassification.from_pretrained(model_path)
    return model, processor

model, processor = load_model()
class_names = ['0Normal', '1Doubtful', '2Mild', '3Moderate', '4Severe']

# --- 2. Streamlit UI ---
st.title("Knee Osteoarthritis ViT Classifier")
st.write("Upload an X-ray image to classify its osteoarthritis severity.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    with st.spinner("Analyzing..."):
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        confidences = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

    st.subheader("Prediction")
    st.bar_chart(confidences)
    top_class = max(confidences, key=confidences.get)
    st.success(f"Most likely: **{top_class}** ({confidences[top_class]:.2%} confidence)")