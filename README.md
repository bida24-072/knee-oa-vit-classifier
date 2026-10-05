# Knee Osteoarthritis Severity Classification using Vision Transformer (ViT)

## 📌 Project Overview
This project focuses on the automated classification of knee osteoarthritis severity from X-ray images. Using a fine-tuned Vision Transformer (ViT), the model classifies images into five categories based on the Kellgren-Lawrence (KL) grading system:
- `0Normal`
- `1Doubtful`
- `2Mild`
- `3Moderate`
- `4Severe`

The project includes a trained model, a live interactive dashboard for testing, and the complete Python code for training and deployment.

## 🚀 Live Dashboard
You can test the model directly in your browser using our interactive Streamlit dashboard:
**[👉 https://knee-oa-vit-classifier-a4mgmdrlbvwdouxbk2my8n.streamlit.app/]**here

*(Note: The current model is a "quick demo" fine-tune trained for 20 steps. It serves as a proof of concept. For production accuracy, the training script can be run for more steps).*

## 🔗 Project Links
- **Trained Model on Hugging Face:** [Theoanoldgaopalelwe/knee-oa-vit-classifier](https://huggingface.co/Theoanoldgaopalelwe/knee-oa-vit-classifier)
- **Dataset:** [knee-arthritis/knee-oa-leakage-free](https://huggingface.co/datasets/knee-arthritis/knee-oa-leakage-free)

## 🛠️ Technical Architecture
- **Base Model:** `google/vit-base-patch16-224-in21k` (Pretrained on ImageNet-21k)
- **Framework:** PyTorch & Hugging Face `transformers`
- **Preprocessing:** 
  - Images are resized to 224x224.
  - **Crucial Fix:** X-ray images are grayscale (1-channel). The code converts them to RGB (3-channel) using `.convert("RGB")` to match the ViT input requirements.
- **Training:** Fine-tuned on the provided dataset using the Hugging Face `Trainer` API.

## ❓ Why Streamlit instead of Hugging Face Spaces?
During the deployment phase, we chose **Streamlit Community Cloud** over Hugging Face Spaces for the following reasons:

1. **Python Execution Requirements:** Hugging Face Spaces offer a "Static" SDK, which only serves HTML/CSS/JavaScript. It cannot execute Python code or run our PyTorch model. 
2. **Hugging Face Pricing Changes:** While Hugging Face offers a "Gradio" SDK that can run Python, it recently moved to a **paid plan** for Gradio Spaces. 
3. **Free & Seamless Integration:** Streamlit Community Cloud provides a completely free, Python-based environment that connects directly to our GitHub repository. This allowed us to deploy our `app.py` without incurring costs, while still providing a live, interactive dashboard for the team.

## 📂 Repository Structure
- `train.py`: The Python script used to fine-tune the ViT model.
- `app.py`: The Streamlit dashboard code for the live web application.
- `requirements.txt`: The Python dependencies needed to run the dashboard.
- `README.md`: Project documentation.

## 💻 How to Run Locally
If you want to run this project on your own machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/knee-oa-vit-classifier.git
   cd knee-oa-vit-classifier
