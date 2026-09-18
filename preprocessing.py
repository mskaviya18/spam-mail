import re
import string
import nltk

# Download necessary NLTK resource quietly
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

def preprocess_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Cleans and preprocesses incoming text for spam classification.
    Preserves key spam signals: URLs, currency symbols, and numbers.
    
    Args:
        text (str): Raw email or message text.
        remove_stopwords (bool): Whether to filter out standard stopwords.
        
    Returns:
        str: Normalized and preprocessed text string.
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Normalize URLs to a standard token while keeping presence clear
    text = re.sub(r'https?://\S+|www\.\S+', ' http_link ', text)

    # Normalize email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' email_address ', text)

    # Standardize whitespace (remove extra spaces, newlines, tabs)
    text = re.sub(r'\s+', ' ', text).strip()

    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words]
        text = ' '.join(tokens)

    return text