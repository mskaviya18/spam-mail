# AI-Powered Spam Email Classifier Using Machine Learning and Groq

An end-to-end Machine Learning web application built using **Streamlit**, **Scikit-Learn**, and **Groq API** to detect spam messages and explain suspicious indicators in natural language.

---

## 📌 Architecture Diagram

Email Input
│
▼
Text Preprocessing ──► TF-IDF Vectorization ──► Scikit-Learn Model
│
▼
Groq AI Explanation ◄────────────────────────────── Prediction & Probability


---

## 🚀 Features

- **Decoupled Classification Logic**: Spam detection runs on local ML algorithms (**Multinomial Naive Bayes / Logistic Regression**), guaranteeing full operation even when offline or without API keys.
- **Natural Language Explanations**: Integrates Groq SDK for real-time safety recommendations and indicator summaries.
- **Real Metrics**: Train-test splits generate actual non-hardcoded evaluation statistics (Accuracy, Precision, Recall, F1).
- **Graceful Fallbacks**: Includes dataset auto-creation and fallback mechanisms if model binaries or secrets are absent.

---

## 🛠️ Local Setup & Running

### 1. Clone Repository & Setup Environment
```bash
git clone [https://github.com/](https://github.com/)<your-username>/spam-email-classifier.git
cd spam-email-classifier
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
2. Configure Local Secrets (Optional for Groq)
Create a file at .streamlit/secrets.toml:

Ini, TOML
GROQ_API_KEY = "gsk_your_actual_groq_api_key"
GROQ_MODEL = "llama-3.3-70b-versatile"
3. Train Model & Run Application
Bash
python train_model.py
streamlit run app.py
☁️ Streamlit Community Cloud Deployment
Push your project to GitHub.

Visit Streamlit Community Cloud.

Connect your repository and specify app.py as the Main file path.

Go to Advanced Settings -> Secrets and paste your secrets:

Ini, TOML
GROQ_API_KEY = "your_actual_groq_api_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"
Click Deploy.


---

# ☁️ Step-by-Step GitHub & Streamlit Cloud Deployment Guide

1. **Create Repository**: Go to GitHub, create a new public repository named `spam-email-classifier`.
2. **Push Files**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Spam Email Classifier"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/spam-email-classifier.git
   git push -u origin main