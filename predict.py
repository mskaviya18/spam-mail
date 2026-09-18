import os
import joblib
import numpy as np
from preprocessing import preprocess_text

MODEL_PATH = os.path.join("model", "spam_classifier.pkl")
VECTORIZER_PATH = os.path.join("model", "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join("model", "metrics.pkl")

class SpamClassifierPredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.metrics = None
        self.load_artifacts()

    def is_ready(self) -> bool:
        """Checks if both model and vectorizer are loaded."""
        return self.model is not None and self.vectorizer is not None

    def load_artifacts(self) -> bool:
        """Loads serialized model, vectorizer, and metrics files."""
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                if os.path.exists(METRICS_PATH):
                    self.metrics = joblib.load(METRICS_PATH)
                return True
            except Exception as e:
                print(f"Error loading model files: {e}")
                self.model = None
                self.vectorizer = None
                return False
        return False

    def predict(self, raw_text: str) -> dict:
        """Classifies incoming text using the loaded ML pipeline."""
        if not self.is_ready():
            raise RuntimeError("Model files not found. Please train the model first.")

        cleaned = preprocess_text(raw_text)
        vec = self.vectorizer.transform([cleaned])
        
        prediction = self.model.predict(vec)[0]

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vec)[0]
            classes = list(self.model.classes_)
            spam_idx = classes.index('spam')
            ham_idx = classes.index('ham')
            spam_prob = float(probs[spam_idx])
            ham_prob = float(probs[ham_idx])
        elif hasattr(self.model, "decision_function"):
            decision = float(self.model.decision_function(vec)[0])
            spam_prob = 1.0 / (1.0 + np.exp(-decision))
            ham_prob = 1.0 - spam_prob
            if prediction == 'ham' and spam_prob > 0.5:
                spam_prob, ham_prob = ham_prob, spam_prob
        else:
            spam_prob = 1.0 if prediction == 'spam' else 0.0
            ham_prob = 1.0 - spam_prob

        confidence = spam_prob if prediction == 'spam' else ham_prob

        return {
            "label": prediction.upper(),
            "is_spam": prediction.lower() == 'spam',
            "confidence": float(confidence),
            "spam_probability": float(spam_prob),
            "ham_probability": float(ham_prob),
            "cleaned_text": cleaned
        }