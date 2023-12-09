from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import CountVectorizer
# from utils import log_prediction
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import torch
from torch import nn
from urllib.parse import quote


app = Flask(__name__)
password = quote(YOUR DB PASSWORD)
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://adhil:{password}@localhost/flaskapp'
db = SQLAlchemy(app)


# logging prediction requests and results in the db
class PredictionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500), nullable=False)
    prediction = db.Column(db.String(10), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# logging predictions to the database
def log_prediction(text, prediction):
    new_prediction_log = PredictionLog(text=text, prediction=prediction)
    db.session.add(new_prediction_log)
    db.session.commit()
    print(f"Logged prediction: Text='{text}', Prediction='{prediction}'")


# Load the pre-trained model and CountVectorizer
model_path = 'model/spam_classifier_model.pth'
loaded_model = torch.load(model_path)
cv = CountVectorizer()

# Result of prediction
def predict_spam(text):
    text_vec = cv.fit_transform([text])
    text_tensor = torch.LongTensor(text_vec.toarray())

    with torch.no_grad():
        model_output = loaded_model(text_tensor)
        _, predicted_label = torch.max(model_output, 1)

        if predicted_label.numpy()[0] == 1:
            result_pred = 'Spam'
        elif predicted_label.numpy()[0] == 0:
            result_pred = 'Ham'
    
    return result_pred


# Define the route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for the prediction_logs
@app.route('/prediction_logs')
def prediction_logs():
    # takes all records from the db table(PredictionLog table)
    prediction_logs = PredictionLog.query.all()
    return render_template('db_logs.html', prediction_logs=prediction_logs)

# Define the route for receiving predictions
@app.route('/predict', methods=['POST'])
def predict():  
    text = request.form['text']
    prediction = predict_spam(text)

    try:
        # Log the prediction in the database
        log_prediction(text, prediction)
    except Exception as e:
        print(f"Error logging prediction: {e}")


    return render_template('index.html', text=text, prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True) 
