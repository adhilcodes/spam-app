import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from model import Net

path = '/home/adhil/Desktop/spam-app/model/data/spam.csv'
data = pd.read_csv(path, encoding = 'latin', usecols = ['v1','v2'])
data.columns = ['label', 'text']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
# print(data.shape)
# data.head()

X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=42)

# Vectorizing the data
cv = CountVectorizer()
X_train_vec = cv.fit_transform(X_train)
X_test_vec = cv.transform(X_test)

# making pytorch tensors
X_train_tensor = torch.LongTensor(X_train_vec.toarray())
X_test_tensor = torch.LongTensor(X_test_vec.toarray())

y_train_tensor = torch.LongTensor(y_train.values)
y_test_tensor = torch.LongTensor(y_test.values)

# print(X_train_tensor.shape, X_test_tensor.shape)
# print(y_train_tensor.shape, y_test_tensor.shape)


input_size = X_train_tensor.shape[1]
hidden_size = 128
output_size = 2 
learning_rate = 0.001
epochs = 500

# Create DataLoader for training data
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = Net(input_size, hidden_size, output_size)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = loss_func(output, batch_y.long())
        loss.backward()
        optimizer.step()
    print(f'epoch {epoch+1}/{epochs}  \t Loss {loss.item()} \n')


# Evaluate on test data
with torch.no_grad():
    model.eval()
    test_output = model(X_test_tensor)
    predicted_labels = torch.argmax(test_output, dim=1)
    accuracy = torch.sum(predicted_labels == y_test_tensor.long()).item() / len(y_test_tensor)
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")


# Save the model
torch.save(model, 'spam_classifier_model.pth')
