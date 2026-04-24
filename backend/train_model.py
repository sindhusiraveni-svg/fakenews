import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from backend .utils import preprocess_input
# 1. Load data - CHANGE THIS PATH if your train.csv is in data/raw/
df = pd.read_csv('data/train.csv') 

# 2. Preprocess text
df['text'] = df['text'].apply(preprocess_input)

# 3. Create features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 4. Train model
model = LogisticRegression()
model.fit(X, y)

# 5. Save model and vectorizer
with open('saved_models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('saved_models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model trained and saved successfully")
print(f"Trained on {len(df)} examples")
