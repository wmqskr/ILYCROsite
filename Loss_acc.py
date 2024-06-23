import torch
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn
from MLP_Attention import MLP
import time
# Read and prepare data
data = pd.read_csv('yourpath')
X = data.drop('label', axis=1).values
y = data['label'].values

# Convert to torch Tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Split dataset and create DataLoader
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = MLP()
import torch.optim as optim

def train_and_evaluate(model, train_loader, test_loader, epochs):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr)
    #scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

            predictions = torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
            correct_train += (predictions == y_batch.unsqueeze(1)).sum().item()
            total_train += y_batch.size(0)

        train_losses.append(train_loss / len(train_loader.dataset))
        train_accuracies.append(correct_train / total_train)

        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1))
                test_loss += loss.item() * X_batch.size(0)
                predictions = torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
                correct_test += (predictions == y_batch.unsqueeze(1)).sum().item()
                total_test += y_batch.size(0)

        test_losses.append(test_loss / len(test_loader.dataset))
        test_accuracies.append(correct_test / total_test)
        #scheduler.step()


        #print(f"Epoch {epoch + 1}, Train Loss: {train_losses[-1]}, Train Accuracy: {train_accuracies[-1]}, Test Loss: {test_losses[-1]}, Test Accuracy: {test_accuracies[-1]}")

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), test_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(range(1, epochs+1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs+1), test_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    # Save model
    torch.save(model, 'com_att_DR2testmodel.pth')
    print("Model saved.")

train_and_evaluate(model, train_loader, test_loader, epochs=200)