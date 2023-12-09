import torch
from torch import nn
from sklearn.feature_extraction.text import CountVectorizer
from model import Net

cv = CountVectorizer()

def test_func(text):
    dd = cv.fit_transform([text])
    dd = torch.LongTensor(dd.toarray())
    return dd

new_data = 'You won 2 crore rupees. please share the OTP with us to claim. You can also make UPI payment to us. Crypto payment also acceptable.'

# Load the modelmodel
model_path = '/home/adhil/Desktop/spam-app/model/spam_classifier_model.pth'
loaded_model = torch.load(model_path)

with torch.no_grad():
    new_data_predictions = loaded_model(test_func(new_data))
    
    _, predicted_labels = torch.max(new_data_predictions, 1)
    spam_probability = torch.softmax(new_data_predictions, 1)[:, 1]
    print('Probability of prediction is: ', spam_probability.item())

    # Display the predicted labels
    # Predicts spam:1 and ham: 0
    if predicted_labels.numpy()[0] == 1:
        print('It is spam')
    elif predicted_labels.numpy()[0] == 0:
        print('It is ham')
    else:
        print('wrong prediction')