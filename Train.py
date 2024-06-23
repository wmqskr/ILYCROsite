import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from MLP_Attention import MLP

# 设备设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 读取并准备数据
data = pd.read_csv('./dataset/train/DR_kcr_cv.csv')
X = data.drop('label', axis=1).values
y = data['label'].values

# 转换为torch的Tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 划分数据集并创建DataLoader
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 创建模型
model = MLP().to(device)

def train_and_evaluate(model, train_loader, test_loader, epochs):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # 保存模型
    torch.save(model.state_dict(),'MLP_ATT_model6.pth')
    print("Model saved.")

train_and_evaluate(model, train_loader, test_loader, epochs=200)
