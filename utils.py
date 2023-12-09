import torch
from torch import nn
from model.model import Net 
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

def load_model(path):
    loaded_model = torch.load(path)
    return loaded_model

def predict_spam(model, cv, text):
    text_vec = cv.fit_transform([text])
    text_tensor = torch.LongTensor(text_vec.toarray())
    
    with torch.no_grad():
        model_output = model(text_tensor)
        _, predicted_label = torch.max(model_output, 1)

        if predicted_label.numpy()[0] == 1:
            result_pred = 'Spam'
        elif predicted_label.numpy()[0] == 0:
             result_pred = 'Ham'
    return result_pred
