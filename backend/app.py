import os
import pickle
from flask import Flask, render_template, request, jsonify

# This makes the path work from any folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__,
            template_folder='../frontend/templates',
            static_folder='../frontend/static')

# Load model using absolute path
model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    news_text = data['news']

    news_vect = vectorizer.transform([news_text]) # Fixed: passes news_text
    prediction = model.predict(news_vect)[0]
    proba = model.predict_proba(news_vect)[0]
    confidence = round(max(proba) * 100, 1)

    return jsonify({
        'prediction': prediction,
        'confidence': confidence,
        'accuracy': 93.2,
        'f1_score': 93.6 # Fixed: underscore not space
    })

if __name__ == '__main__':
    port=int(os.environ.get('port',5000))
    app.run(host='0.0.0.0',port=port,debug=False)
    

    



    
    
