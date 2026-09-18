import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from preprocessing import preprocess_text

DATA_PATH = os.path.join("data", "spam.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "spam_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.pkl")

FALLBACK_DATASET = [
    ("spam", "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim http://bit.ly/claim-now"),
    ("spam", "URGENT! Your bank account has been suspended. Verify your credentials immediately at http://secure-bank.com"),
    ("spam", "Get cheap loans now with 0% interest. Reply YES to get approved in 5 minutes!"),
    ("spam", "Earn $5,000 working from home per week! No experience needed. Contact admin@job-fast.com"),
    ("spam", "You have 1 unread message from your secret admirer. Call 1-900-555-0199 now!"),
    ("spam", "Exclusive deal! Buy 1 Rolex get 1 free! Visit http://watch-deals.xyz"),
    ("spam", "Claim your free Bitcoin bonus today! Deposit $10 and get $500 back instantly."),
    ("spam", "Final notice: Tax refund pending of $2,450. Submit your SSN and card details at http://irs-refund-portal.org"),
    ("ham", "Hi John, can we schedule our project sync meeting for tomorrow at 10 AM?"),
    ("ham", "Hey, are we still meeting for lunch today at the cafeteria? Let me know."),
    ("ham", "Please review the attached quarterly report and let me know your thoughts before Friday."),
    ("ham", "Don't forget to buy milk and bread on your way back home."),
    ("ham", "Dear student, your assignment submission has been successfully received by the portal."),
    ("ham", "Thanks for sending over the updated code. I'll test it and push to main branch."),
    ("ham", "Can you send me the slides from today's lecture? I missed the first half."),
    ("ham", "Your doctor's appointment is confirmed for Tuesday at 3 PM. Reply C to cancel.")
]

def load_dataset() -> pd.DataFrame:
    """Loads dataset from disk or builds a fallback dataset if missing."""
    os.makedirs("data", exist_ok=True)
    
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH, encoding='latin-1')
            # Handle standard SMS Spam Collection formats or custom CSVs
            if 'v1' in df.columns and 'v2' in df.columns:
                df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
            elif 'Category' in df.columns and 'Message' in df.columns:
                df = df[['Category', 'Message']].rename(columns={'Category': 'label', 'Message': 'text'})
            elif 'label' in df.columns and 'text' in df.columns:
                df = df[['label', 'text']]
            else:
                raise ValueError("CSV schema unrecognized.")
            
            # Convert numeric labels if present
            if df['label'].dtype in [int, np.int64]:
                df['label'] = df['label'].map({1: 'spam', 0: 'ham'})
            df['label'] = df['label'].str.lower().str.strip()
            return df.dropna()
        except Exception as e:
            print(f"Warning: Failed to load {DATA_PATH} ({e}). Falling back to built-in dataset.")

    df = pd.DataFrame(FALLBACK_DATASET, columns=['label', 'text'])
    df.to_csv(DATA_PATH, index=False)
    return df

def train_and_evaluate():
    """Trains TF-IDF + Classifier models and evaluates real performance metrics."""
    print("Loading data...")
    df = load_dataset()

    print(f"Dataset size: {len(df)} rows")
    print(df['label'].value_counts())

    # Preprocess text column
    df['cleaned_text'] = df['text'].apply(lambda x: preprocess_text(str(x)))

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Candidate models
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Linear SVM": LinearSVC(C=1.0)
    }

    results = {}
    best_model = None
    best_f1 = -1.0
    best_name = ""

    print("\n--- Model Comparison ---")
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, pos_label='spam', zero_division=0)
        rec = recall_score(y_test, preds, pos_label='spam', zero_division=0)
        f1 = f1_score(y_test, preds, pos_label='spam', zero_division=0)
        cm = confusion_matrix(y_test, preds, labels=['ham', 'spam'])

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": cm.tolist()
        }

        print(f"Model: {name}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}\n")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    print(f"Selected Best Model: {best_name} (F1-Score: {best_f1:.4f})")

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    # Store complete metadata and evaluation metrics
    metrics_payload = {
        "best_model_name": best_name,
        "all_results": results,
        "test_counts": y_test.value_counts().to_dict(),
        "train_size": len(X_train),
        "test_size": len(X_test)
    }
    joblib.dump(metrics_payload, METRICS_PATH)
    print("Model, Vectorizer, and Metrics saved successfully!")

if __name__ == "__main__":
    train_and_evaluate()