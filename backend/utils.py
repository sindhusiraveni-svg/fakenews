def clean_text(text):
    """
    For BERT models: do minimal cleaning. 
    BERT was trained on raw text with punctuation and case.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Only remove extra spaces/newlines. Keep punctuation!
    text = " ".join(text.split())
    text = text.strip()
    return text

def preprocess_input(text):
    """
    Main preprocessing function used by predict.py
    """
    cleaned = clean_text(text)
    return cleaned

