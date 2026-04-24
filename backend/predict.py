from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the BERT model you trained
model_path = "saved_models/bert_model" # or wherever you saved it
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
def predict_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    print("DEBUG:", probs)
    
    fake_prob = probs[0][0].item() # 0=Fake for most datasets
    real_prob = probs[0][1].item() # 1=Real for most datasets
    return {"Fake": round(fake_prob, 4), "Real": round(real_prob, 4)}



    


