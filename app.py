import os
import streamlit as st
import pandas as pd
from predict import SpamClassifierPredictor
from train_model import train_and_evaluate

# Page configuration
st.set_page_config(
    page_title="AI Spam Email Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Predictor & Auto-train if artifacts are missing
predictor = SpamClassifierPredictor()
if not predictor.is_ready():
    with st.spinner("Initializing ML model files for the first time..."):
        train_and_evaluate()
        predictor.load_artifacts()

# Function to query Groq safely
def get_groq_explanation(email_text: str, ml_label: str, confidence: float) -> tuple[str, str]:
    """
    Fetches explanation from Groq API without exposing API keys or crashing on failure.
    Returns tuple: (explanation_text, status_message)
    """
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    groq_model = st.secrets.get("GROQ_MODEL") or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    if not groq_api_key:
        return "", "ℹ️ AI explanation unavailable (GROQ_API_KEY missing). ML classification is still working."

    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)

        prompt = f"""
You are a cybersecurity expert analyzing an email flag by a Machine Learning model.

Email Content:
\"\"\"{email_text[:1500]}\"\"\"

ML Model Result: {ml_label} (Confidence: {confidence*100:.1f}%)

Provide a concise analysis using strictly this layout:

### AI Explanation

**Why:**
[1-2 sentences explaining why it appears suspicious or legitimate]

**Indicators:**
* [Key factor 1]
* [Key factor 2]
* [Key factor 3]

**Recommendation:**
[1-2 sentence practical advice for the recipient]

Do not challenge or override the ML classification. Focus solely on explaining the indicators present.
"""

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful cybersecurity analyst."},
                {"role": "user", "content": prompt}
            ],
            model=groq_model,
            temperature=0.2,
            max_tokens=400
        )
        return response.choices[0].message.content, ""

    except Exception as e:
        return "", f"⚠️ Could not fetch AI explanation: {str(e)}"


# Sidebar Navigation
st.sidebar.title("📧 Navigation")
page = st.sidebar.radio(
    "Select Section", 
    ["🏠 Home", "📊 Model Performance", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**System Status**")
st.sidebar.success("ML Model: Online")
if st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY"):
    st.sidebar.success("Groq AI: Configured")
else:
    st.sidebar.warning("Groq AI: Key missing")

# Main Header
st.title("📧 AI Spam Email Classifier")
st.caption("Detect suspicious emails using Machine Learning + NLP with Groq Insights")

# PAGE 1: HOME
if page == "🏠 Home":
    st.markdown("### Paste email or test sample")
    
    # Example dataset selection
    examples = {
        "Custom Input": "",
        "Spam Example 1: Fake Bank Urgent Notice": "URGENT: Your account at Chase Bank has been temporarily locked due to suspicious activity. Verify your identity immediately at http://verify-chase-update.com or your funds will be frozen.",
        "Spam Example 2: Lottery Win": "Congratulations! You've been chosen as the lucky winner of $1,000,000 in our international promo! Reply with your phone number and address to claim your prize.",
        "Spam Example 3: Fake Job Offer": "Make $500/day working 1 hour from home! No experience necessary. Payment processed daily. Register here: http://quick-cash-now.xyz",
        "Spam Example 4: Phishing Password Reset": "Your Microsoft Office 365 password expires today. Keep your current password by verifying here http://msoffice-renewal.net",
        "Spam Example 5: Crypto Scheme": "Guaranteed 300% return on Bitcoin in 24 hours! Limited slots left. Send BTC to our wallet now.",
        "Ham Example 1: Meeting Request": "Hi Sarah, let's reschedule our code review meeting to Thursday at 2 PM. Let me know if that time works for you.",
        "Ham Example 2: Project Delivery": "Hi Team, I have pushed the latest updates to the development branch. Please review the pull request when you have time.",
        "Ham Example 3: Grocery List": "Hey, don't forget to grab eggs, milk, and coffee beans on your way home tonight.",
        "Ham Example 4: University Notification": "Dear Student, your semester grades have been finalized and uploaded to the student portal.",
        "Ham Example 5: Flight Confirmation": "Your flight booking #FL9823 to San Francisco is confirmed. Check-in opens 24 hours prior to departure."
    }

    selected_example = st.selectbox("💡 Try a sample message:", list(examples.keys()))
    
    default_text = examples[selected_example] if selected_example != "Custom Input" else ""
    user_input = st.text_area("Email Content", value=default_text, height=180, placeholder="Paste message or email body here...")

    col1, col2 = st.columns([1, 4])
    with col1:
        classify_button = st.button("🔍 Classify Email", use_container_width=True)

    if classify_button or user_input:
        if not user_input.strip():
            st.warning("Please enter email text to perform classification.")
        else:
            try:
                res = predictor.predict(user_input)
                label = res["label"]
                confidence = res["confidence"]
                spam_prob = res["spam_probability"]
                ham_prob = res["ham_probability"]

                st.markdown("---")
                st.subheader("Classification Result")

                # Display Main Badge
                if res["is_spam"]:
                    st.error("🚨 **SPAM DETECTED**")
                else:
                    st.success("✅ **NOT SPAM (HAM)**")

                # Display Metrics Metrics
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Prediction", label)
                m_col2.metric("Confidence", f"{confidence * 100:.2f}%")
                m_col3.metric("Spam Probability", f"{spam_prob * 100:.2f}%")

                # Probability visual bar
                st.write("**Probability Distribution**")
                st.progress(spam_prob, text=f"Spam: {spam_prob*100:.1f}% | Legitimate (Ham): {ham_prob*100:.1f}%")

                # Groq Explanation Block
                st.markdown("---")
                st.subheader("🤖 AI Explanation & Insights")
                
                with st.spinner("Requesting explanation from Groq AI..."):
                    explanation, err_msg = get_groq_explanation(user_input, label, confidence)

                if explanation:
                    st.markdown(explanation)
                elif err_msg:
                    st.info(err_msg)

            except Exception as e:
                st.error(f"Error during classification: {str(e)}")

# PAGE 2: MODEL PERFORMANCE
elif page == "📊 Model Performance":
    st.subheader("Model Evaluation & Statistics")
    
    if predictor.metrics:
        metrics = predictor.metrics
        best_model_name = metrics.get("best_model_name", "Primary Classifier")
        results = metrics.get("all_results", {})

        st.info(f"Active Selected Model: **{best_model_name}**")

        if best_model_name in results:
            res = results[best_model_name]
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{res['accuracy']*100:.2f}%")
            c2.metric("Precision", f"{res['precision']*100:.2f}%")
            c3.metric("Recall", f"{res['recall']*100:.2f}%")
            c4.metric("F1-Score", f"{res['f1_score']*100:.2f}%")

            st.markdown("### Confusion Matrix")
            cm = res['confusion_matrix']
            cm_df = pd.DataFrame(
                cm, 
                index=["Actual Ham", "Actual Spam"], 
                columns=["Predicted Ham", "Predicted Spam"]
            )
            st.dataframe(cm_df, use_container_width=True)

        st.markdown("### All Tested Models Comparison")
        comp_data = []
        for model_name, m_val in results.items():
            comp_data.append({
                "Model": model_name,
                "Accuracy": f"{m_val['accuracy']*100:.2f}%",
                "Precision": f"{m_val['precision']*100:.2f}%",
                "Recall": f"{m_val['recall']*100:.2f}%",
                "F1-Score": f"{m_val['f1_score']*100:.2f}%"
            })
        st.table(pd.DataFrame(comp_data))

    else:
        st.warning("Metrics file not found. Re-run model training to generate complete validation stats.")

    if st.button("🔄 Retrain Model Now"):
        with st.spinner("Retraining model on dataset..."):
            train_and_evaluate()
            predictor.load_artifacts()
        st.success("Model retrained successfully!")
        st.rerun()

# PAGE 3: ABOUT
elif page == "ℹ️ About":
    st.subheader("System Architecture & Implementation Details")
    st.markdown("""
    ### Machine Learning Pipeline
    1. **Preprocessing**: Normalizes email input, lowers casing, maps URLs and email tokens, removes extraneous spaces.
    2. **TF-IDF Vectorization**: Converts raw text strings into numerical sublinear term frequency vectors (using unigrams and bigrams).
    3. **Classification**: Uses Naive Bayes / Logistic Regression trained on ground-truth labeled email data to output probabilities.
    4. **Groq AI Layer**: Generates human-readable cybersecurity contextual explanations for suspicious indicators without changing ML decision outputs.

    ### Privacy & Security
    * Emails are processed strictly in-memory during inference.
    * No user input or API secrets are stored permanently or logged.
    """)