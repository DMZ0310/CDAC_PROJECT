import os, torch, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Dataset (Cached Features)
# =========================
class FeatureDataset(Dataset):
    def __init__(self, csv_path, feature_dir):
        df = pd.read_csv(csv_path)
        self.files = df["file"].tolist()
        self.labels = df["label"].tolist()

        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.labels)
        self.feature_dir = feature_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        feat = torch.load(
            os.path.join(self.feature_dir, self.files[idx].replace(".wav", ".pt"))
        )
        return feat, torch.tensor(self.labels[idx])

def collate_fn(batch):
    feats, labels = zip(*batch)
    feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    return feats.to(DEVICE), torch.tensor(labels).to(DEVICE)

# =========================
# Model
# =========================
class CNN_BiLSTM(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(768, 128, 3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(256, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.permute(0,2,1)
        x,_ = self.lstm(x)
        return self.fc(x.mean(dim=1))

# =========================
# Train + Evaluate
# =========================
def train():
    ds = FeatureDataset("transcripts.csv", "features")
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn)

    model = CNN_BiLSTM(len(ds.le.classes_)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    train_acc, train_loss = [], []

    for epoch in range(10):
        correct = total = loss_sum = 0
        y_true, y_pred = [], []

        for x,y in tqdm(dl, desc=f"Epoch {epoch+1}"):
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out,y)
            loss.backward()
            opt.step()

            loss_sum += loss.item()
            preds = out.argmax(1)
            correct += (preds==y).sum().item()
            total += y.size(0)

            y_true.extend(y.cpu())
            y_pred.extend(preds.cpu())

        acc = correct/total
        train_acc.append(acc)
        train_loss.append(loss_sum)

        print(f"Epoch {epoch+1} | Acc: {acc:.4f}")

    # ===== Confusion Matrix =====
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=ds.le.classes_)
    disp.plot(cmap="Blues")
    plt.savefig("plots/confusion_matrix.png")

    # ===== Accuracy & Loss =====
    plt.figure()
    plt.plot(train_acc, label="Accuracy")
    plt.legend()
    plt.savefig("plots/accuracy.png")

    plt.figure()
    plt.plot(train_loss, label="Loss")
    plt.legend()
    plt.savefig("plots/loss.png")

    torch.save(model.state_dict(), "final_model.pt")

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    train()
