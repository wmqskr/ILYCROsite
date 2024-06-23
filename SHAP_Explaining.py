import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F

data = pd.read_csv('./dataset_test/AAC_testData.csv')
X = data.drop('label', axis=1).values
y = data['label'].values

# Convert to torch Tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Split dataset and create DataLoader
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class Attention(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(in_features, hidden_dim)
        self.key = nn.Linear(in_features, hidden_dim)
        self.value = nn.Linear(in_features, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(query.shape[-1]).float())
        attention_weights = self.softmax(attention_scores)
        attention_output = torch.matmul(attention_weights, value)
        return attention_output

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(20, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.output = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.46)
        self.attention = Attention(64, 64)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.attention(x)
        x = torch.sigmoid(self.output(x))
        return x

model = MLP()
# Load saved model
model.load_state_dict(torch.load('com_att_AAC0model.pth'))
model.eval()

import shap

# Define a wrapper for the model's forward pass for SHAP
class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            tensor_X = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(tensor_X)
            return outputs.detach().numpy()

# Use a smaller background dataset
background = X_train.numpy()[:10]

# Use Explainer
explainer = shap.Explainer(ModelWrapper(model).predict, background)

# Prepare test data
sample_test = X_test.numpy()[:3500]

features = ['A', 'C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
# Calculate SHAP values
shap_values = explainer.shap_values(sample_test)

# Plot SHAP values for multiple samples using summary_plot
shap.summary_plot(shap_values, sample_test, feature_names=features)
shap.dependence_plot('P', shap_values, sample_test, feature_names=features)