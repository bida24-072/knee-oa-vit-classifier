import streamlit as st
import torch
import pandas as pd
import altair as alt
import requests
from io import BytesIO
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
    st.info("This dashboard uses a fine-tuned **Vision Transformer (ViT)** to classify knee osteoarthritis severity.")
    st.warning("⚠️ **Proof of Concept:** Not medically accurate yet.")
    st.markdown("---")
    st.write("**Model:** [Theoanoldgaopalelwe/knee-oa-vit-classifier](https://huggingface.co/Theoanoldgaopalelwe/knee-oa-vit-classifier)")

# --- 5. Main UI ---
st.title("🦴 Knee Osteoarthritis ViT Classifier")

# Create Tabs
tab1, tab2 = st.tabs(["📷 Single Image Prediction", "📁 Batch Inference"])

# ==========================================
# TAB 1: Single Image Prediction
# ==========================================
with tab1:
    st.write("Upload, paste a screenshot (Ctrl+V), or paste a URL of a knee X-ray.")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("1. Provide Image")
        uploaded_file = st.file_uploader("Upload or paste (Ctrl+V) an X-ray image", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="single_uploader")
        st.write("**OR**")
        image_url = st.text_input("Paste an Image URL")

        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        elif image_url:
            try:
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                st.image(image, caption="Image from URL", use_column_width=True)
            except Exception as e:
                st.error(f"Could not load image from URL. Error: {e}")

    with col2:
        st.subheader("2. Analysis Results")
        if image is None:
            st.info("Awaiting image... Upload a file, paste a screenshot, or provide a URL.")
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
                st.dataframe(
                    df,
                    column_config={
                        "Severity": st.column_config.TextColumn("Severity"),
                        "Confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            format="percent",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                    },
                    use_container_width=True,
                    hide_index=True
                )

# ==========================================
# TAB 2: Batch Inference
# ==========================================
with tab2:
    st.write("Run batch inference by uploading multiple local images, or by using a Hugging Face dataset ID.")
    
    source = st.radio("Select Data Source:", ["📤 Upload Local Images", "🤗 Use Hugging Face Dataset"], horizontal=True)
    
    st.markdown("---")

    # --- Option A: Upload Local Images ---
    if source == "📤 Upload Local Images":
        st.info("⚠️ **Note:** To prevent the app from crashing, we limit processing to a maximum of **20 images** per batch.")
        
        uploaded_files = st.file_uploader(
            "Upload a batch of X-ray images", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True,
            key="batch_uploader"
        )
        
        if st.button("🚀 Run Batch Inference on Uploaded Images"):
            if not uploaded_files:
                st.warning("Please upload at least one image.")
            else:
                num_files = min(len(uploaded_files), 20)
                if len(uploaded_files) > 20:
                    st.warning(f"You uploaded {len(uploaded_files)} images. Processing only the first 20 to prevent memory crash.")
                
                results = []
                progress_bar = st.progress(0)
                
                for i, file in enumerate(uploaded_files[:20]):
                    img = Image.open(file).convert("RGB")
                    
                    # Run CLIP Bouncer
                    texts = ["a medical knee X-ray", "a photograph of a person", "an animal", "a random object or scene"]
                    clip_inputs = clip_processor(text=texts, images=img, return_tensors="pt", padding=True)
                    with torch.no_grad():
                        clip_outputs = clip_model(**clip_inputs)
                        clip_probs = clip_outputs.logits_per_image.softmax(dim=1)[0]
                    
                    best_match_idx = clip_probs.argmax().item()
                    best_match_label = texts[best_match_idx]
                    clip_conf = clip_probs[best_match_idx].item()

                    if best_match_label != "a medical knee X-ray" or clip_conf < 0.4:
                        # Invalid image
                        results.append({
                            "Image": img,
                            "Filename": file.name,
                            "Status": f"❌ Invalid: {best_match_label}",
                            "Predicted": "N/A",
                            "Confidence": 0.0  # Numeric for ProgressColumn
                        })
                    else:
                        # Valid image, run ViT
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
                            "Filename": file.name,
                            "Status": "✅ Valid",
                            "Predicted": pred_label,
                            "Confidence": conf
                        })
                    
                    progress_bar.progress((i + 1) / num_files)
                
                st.markdown("### 📋 Batch Results")
                results_df = pd.DataFrame(results)
                st.dataframe(
                    results_df,
                    column_config={
                        "Image": st.column_config.ImageColumn("X-ray", width="small"),
                        "Filename": st.column_config.TextColumn("Filename"),
                        "Status": st.column_config.TextColumn("Status"),
                        "Predicted": st.column_config.TextColumn("Predicted"),
                        "Confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            format="percent",
                            min_value=0.0,
                            max_value=1.0,
                        ),
                    },
                    use_container_width=True,
                    hide_index=True
                )
                st.balloons()

    # --- Option B: Use Hugging Face Dataset ---
    else:
        st.info("⚠️ **Note:** To prevent the app from crashing, we limit processing to a maximum of **10 images** per batch.")
        dataset_id = st.text_input("Hugging Face Dataset ID", value="knee-arthritis/knee-oa-leakage-free")
        split = st.selectbox("Dataset Split", ["test", "validation", "train"], index=0)
        num_samples = st.slider("Number of images to process", min_value=1, max_value=10, value=5)
        
        if st.button("🚀 Run Batch Inference on HF Dataset"):
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
                            "Confidence": conf
                        })
                        
                        progress_bar.progress((i + 1) / num_samples)
                    
                    st.markdown("### 📋 Batch Results")
                    results_df = pd.DataFrame(results)
                    st.dataframe(
                        results_df,
                        column_config={
                            "Image": st.column_config.ImageColumn("X-ray", width="medium"),
                            "True Label": st.column_config.TextColumn("True Label"),
                            "Predicted": st.column_config.TextColumn("Predicted"),
                            "Confidence": st.column_config.ProgressColumn(
                                "Confidence",
                                format="percent",
                                min_value=0.0,
                                max_value=1.0,
                            ),
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error loading dataset: {e}")
                    st.info("Please ensure the Dataset ID is correct and is public.")

st.markdown("---")
st.caption("Note: This is a proof-of-concept model. Always consult a medical professional for diagnosis.")