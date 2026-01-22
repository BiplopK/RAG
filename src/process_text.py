import re 

def preprocess_text(text):
    """Remove extra spaces and clean text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()