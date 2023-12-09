from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from utils import load_model, predict_spam

app = Flask(__name__)

# Load the pre-trained model and CountVectorizer
model_path = 'model/spam_classifier_model.pth'
loaded_model = load_model(model_path)
cv = CountVectorizer()

# Define the route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for receiving predictions
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = predict_spam(loaded_model, cv, text)
    return render_template('index.html', text=text, prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
