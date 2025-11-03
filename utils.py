import re
import nltk
from nltk.corpus import stopwords

# Ensure NLTK resources are available when running the script for the first time.
def ensure_nltk():
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except:
        nltk.download("stopwords")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # basic cleaning
    text = text.lower()
    text = re.sub(r"http\S+","", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text