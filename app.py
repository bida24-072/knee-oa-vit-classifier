import streamlit as st
import torch
from transformers import ViTForImageClassification, ViTImageProcessor, CLIPModel, CLIPProcessor
from PIL import Image

# --- 1. Page Config (Must be the first Streamlit command) ---
st.set_page_config(page_title="Knee OA ViT Classifier", page_icon="🦴", layout="wide")

# --- 2. Custom CSS for a Professional Medical Theme ---
st.markdown("""
    <style>
    /* Main background and font */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    /* Headers */
    h1, h2, h3 {
        color: #4DB6AC !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* File uploader box */
    [data-testid="stFileUploader"] {
        background-color: #1E2129;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4DB6AC;
    }
    /* Success and Error boxes */
    .stAlert {
        border-radius: 10px;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161A23;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Load Models (Cached) ---
@st.cache_resource
def load_vit_model():
    model_path = "Theoanoldgaopalelwe/knee-oa-vit-classifier"
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTForImageClassification.from_pretrained(model_path)
    return model, processor

@st.cache_resource
def load_clip_model():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor

vit_model, vit_processor = load_vit_model()
clip_model, clip_processor = load_clip_model()

class_names = ['0Normal', '1Doubtful', '2Mild', '3Moderate', '4Severe']

# --- 4. Sidebar Info ---
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=50)
    st.title("About This Project")
    st.info(
        "This dashboard uses a fine-tuned **Vision Transformer (ViT)** to classify "
        "knee osteoarthritis severity from X-rays based on the KL grading system."
    )
    st.warning(
        "⚠️ **Proof of Concept:** This model was trained for 20 steps. "
        "It is not medically accurate yet."
    )
    st.markdown("---")
    st.write("**Model:** [Theoanoldgaopalelwe/knee-oa-vit-classifier](https://huggingface.co/Theoanoldgaopalelwe/knee-oa-vit-classifier)")

# --- 5. Main UI Layout ---
st.title("🦴 Knee Osteoarthritis ViT Classifier")
st.write("Upload a knee X-ray image to classify its osteoarthritis severity (KL Grades 0-4).")

# Create two columns: one for the image, one for the results
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("2. Analysis Results")
    
    if uploaded_file is None:
        st.info("Awaiting image upload...")
    else:
        # --- CLIP Bouncer Logic ---
        with st.spinner("Validating image..."):
            texts = ["a medical knee X-ray", "a photograph of a person", "an animal", "a random object or scene"]
            inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)[0]
                
            best_match_idx = probs.argmax().item()
            best_match_label = texts[best_match_idx]
            confidence = probs[best_match_idx].item()

        if best_match_label != "a medical knee X-ray" or confidence < 0.4:
            st.error(f"❌ **Invalid Image Detected**\n\nThis looks like **{best_match_label}**. Please upload a valid knee X-ray.")
        else:
            # --- ViT Inference Logic ---
            with st.spinner("Analyzing X-ray..."):
                vit_inputs = vit_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    vit_outputs = vit_model(**vit_inputs)
                    logits = vit_outputs.logits

                probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
                confidences = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

            max_confidence = max(confidences.values())
            top_class = max(confidences, key=confidences.get)

            # Display Results
            st.success(f"**Prediction:** {top_class} ({max_confidence:.2%} confidence)")
            st.bar_chart(confidences)
            
            st.markdown("---")
            st.caption("Note: This is a proof-of-concept model. Always consult a medical professional for diagnosis.")