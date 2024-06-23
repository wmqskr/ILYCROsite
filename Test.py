import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from MLP_Attention import MLP, Attention

# 设备设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = MLP().to(device)
model.load_state_dict(torch.load('MLP_ATT_model6.pth', map_location=device))
model.eval()

# 替换为你的csv文件名和路径
test_data_path = './dataset/test/DR_Kcr_IND.csv'
test_data = pd.read_csv(test_data_path)

# 假定最后一列为标签
X_test = torch.tensor(test_data.iloc[:, :-1].values, dtype=torch.float32)
y_test = torch.tensor(test_data.iloc[:, -1].values, dtype=torch.float32)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

y_true = []
y_pred = []
y_pred_proba = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        outputs = model(X_batch).squeeze()
        preds = torch.round(outputs)  # 预测标签
        y_true.extend(y_batch.tolist())
        y_pred.extend(preds.tolist())
        y_pred_proba.extend(outputs.tolist())

# 计算性能指标
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
tn_t, fp_t, fn_t, tp_t = torch.tensor(tn), torch.tensor(fp), torch.tensor(fn), torch.tensor(tp)
sn = tp / (tp + fn)
sp = tn / (tn + fp)
acc = (tp + tn) / (tp + tn + fp + fn)
mcc = ((tp_t * tn_t) - (fp_t * fn_t)) / torch.sqrt((tp_t + fp_t) * (tp_t + fn_t) * (tn_t + fp_t) * (tn_t + fn_t))
mcc = mcc.item() if mcc.numel() == 1 else mcc
auc = roc_auc_score(y_true, y_pred_proba)

print(f'SN: {sn:.4f}, SP: {sp:.4f}, ACC: {acc:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}')