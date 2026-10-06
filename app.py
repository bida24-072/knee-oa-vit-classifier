import streamlit as st
import torch
import pandas as pd
import altair as alt
from transformers import ViTForImageClassification, ViTImageProcessor, CLIPModel, CLIPProcessor
from PIL import Image
from datasets import load_dataset

# --- 1. Page Config ---
st.set_page_config(page_title="Knee OA ViT Classifier", page_icon="🦴", layout="wide")

# --- 2. Custom CSS for a Professional Medical Theme ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    h1, h2, h3 { color: #4DB6AC !important; font-family: 'Helvetica Neue', sans-serif; }
    [data-testid="stFileUploader"] {
        background-color: #1E2129; padding: 20px; border-radius: 10px; border: 1px solid #4DB6AC;
    }
    .stAlert { border-radius: 10px; }
    [data-testid="stSidebar"] { background-color: #161A23; }
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
    st.warning("⚠️ **Proof of Concept:** This model was trained for 20 steps. It is not medically accurate yet.")
    st.markdown("---")
    st.write("**Model:** [Theoanoldgaopalelwe/knee-oa-vit-classifier](https://huggingface.co/Theoanoldgaopalelwe/knee-oa-vit-classifier)")

# --- 5. Main UI ---
st.title("🦴 Knee Osteoarthritis ViT Classifier")

# Create Tabs
tab1, tab2 = st.tabs(["📷 Single Image Prediction", "📁 Dataset Inference"])

# ==========================================
# TAB 1: Single Image Prediction
# ==========================================
with tab1:
    st.write("Upload a knee X-ray image to classify its osteoarthritis severity (KL Grades 0-4).")
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
                with st.spinner("Analyzing X-ray..."):
                    vit_inputs = vit_processor(images=image, return_tensors="pt")
                    with torch.no_grad():
                        vit_outputs = vit_model(**vit_inputs)
                        logits = vit_outputs.logits
                    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
                    confidences = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

                max_confidence = max(confidences.values())
                top_class = max(confidences, key=confidences.get)

                df = pd.DataFrame(list(confidences.items()), columns=['Severity', 'Confidence'])
                
                st.markdown("### 🎯 Top Prediction")
                st.metric(label="Predicted Severity", value=top_class, delta=f"{max_confidence:.2%} confidence")

                st.markdown("### 📊 Probability Bar Chart")
                bar_chart = alt.Chart(df).mark_bar().encode(
                    y=alt.Y('Severity', sort='-x', axis=alt.Axis(title='Severity')),
                    x=alt.X('Confidence', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(title='Probability')),
                    color=alt.Color('Severity', legend=None),
                    tooltip=['Severity', alt.Tooltip('Confidence', format='.2%')]
                ).properties(height=250)
                st.altair_chart(bar_chart, use_container_width=True)

                st.markdown("### 🍩 Probability Distribution")
                donut_chart = alt.Chart(df).mark_arc(innerRadius=60).encode(
                    theta=alt.Theta(field="Confidence", type="quantitative"),
                    color=alt.Color(field="Severity", type="nominal"),
                    tooltip=['Severity', alt.Tooltip('Confidence', format='.2%')]
                ).properties(height=300)
                st.altair_chart(donut_chart, use_container_width=True)

                st.markdown("### 📋 Detailed Breakdown")
                styled_df = df.style.background_gradient(subset=['Confidence'], cmap='Teal')
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

# ==========================================
# TAB 2: Dataset Inference
# ==========================================
with tab2:
    st.write("Paste a Hugging Face dataset ID to run batch inference on a small subset of images.")
    st.info("⚠️ **Note:** To prevent the app from crashing, we limit processing to a maximum of 10 images.")
    
    dataset_id = st.text_input("Hugging Face Dataset ID", value="knee-arthritis/knee-oa-leakage-free")
    split = st.selectbox("Dataset Split", ["test", "validation", "train"], index=0)
    num_samples = st.slider("Number of images to process", min_value=1, max_value=10, value=5)
    
    if st.button("🚀 Run Batch Inference"):
        if not dataset_id:
            st.warning("Please enter a valid Dataset ID.")
        else:
            try:
                with st.spinner(f"Loading dataset '{dataset_id}' ({split} split)..."):
                    ds = load_dataset(dataset_id, split=split)
                
                st.success(f"Loaded {len(ds)} images. Processing first {num_samples}...")
                
                results = []
                progress_bar = st.progress(0)
                
                for i in range(min(num_samples, len(ds))):
                    sample = ds[i]
                    img = sample['image'].convert("RGB")
                    true_label_idx = sample['label']
                    true_label = class_names[true_label_idx] if true_label_idx < len(class_names) else f"Unknown ({true_label_idx})"
                    
                    # ViT Inference
                    vit_inputs = vit_processor(images=img, return_tensors="pt")
                    with torch.no_grad():
                        vit_outputs = vit_model(**vit_inputs)
                        logits = vit_outputs.logits
                    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
                    pred_idx = probabilities.argmax().item()
                    pred_label = class_names[pred_idx]
                    conf = probabilities[pred_idx].item()
                    
                    results.append({
                        "Image": img,
                        "True Label": true_label,
                        "Predicted": pred_label,
                        "Confidence": f"{conf:.2%}"
                    })
                    
                    progress_bar.progress((i + 1) / num_samples)
                
                st.markdown("### 📋 Batch Results")
                results_df = pd.DataFrame(results)
                
                # Display results in a nice table with images
                st.dataframe(
                    results_df,
                    column_config={
                        "Image": st.column_config.ImageColumn("X-ray", width="medium"),
                        "True Label": st.column_config.TextColumn("True Label"),
                        "Predicted": st.column_config.TextColumn("Predicted"),
                        "Confidence": st.column_config.TextColumn("Confidence"),
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error loading dataset: {e}")
                st.info("Please ensure the Dataset ID is correct (e.g., 'knee-arthritis/knee-oa-leakage-free') and is public.")

st.markdown("---")
st.caption("Note: This is a proof-of-concept model. Always consult a medical professional for diagnosis.")