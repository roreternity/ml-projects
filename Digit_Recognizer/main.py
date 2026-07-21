import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class MNISTDataset(Dataset):
    def __init__(self, df, is_test=False):
        self.is_test = is_test
        if not is_test:
            self.labels = df["label"].values
            # Нормализуем пиксели в диапазон [0, 1]
            self.features = df.drop("label", axis=1).values.astype("float32") / 255.0
        else:
            self.features = df.values.astype("float32") / 255.0

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Conv2d требует канал цвета: (1, 28, 28), а не просто (28, 28)
        image = torch.tensor(self.features[idx], dtype=torch.float32).view(1, 28, 28)
        if not self.is_test:
            return image, torch.tensor(self.labels[idx], dtype=torch.long)
        return image


# Размечаем train.csv на train/val сами — иначе модель "валидировалась" бы
# на тестовом файле Kaggle, у которого нет правильных ответов
full_train_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
train_df, val_df = train_test_split(full_train_df, test_size=0.2, random_state=42)
final_test_df = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

train_loader = DataLoader(MNISTDataset(train_df), batch_size=64, shuffle=True)
val_loader = DataLoader(MNISTDataset(val_df), batch_size=64, shuffle=False)
test_loader = DataLoader(MNISTDataset(final_test_df, is_test=True), batch_size=64, shuffle=False)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# 'mps' на Apple Silicon, 'cuda' на Kaggle/Nvidia, иначе 'cpu'
device = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(f"Обучение на устройстве: {device}")

epochs = 5
for epoch in range(epochs):
    model.train()
    train_loss, correct_train, total_train = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_train_loss = train_loss / total_train
    epoch_train_acc = correct_train / total_train

    # На валидации веса не обновляем — только смотрим метрики,
    # чтобы вовремя заметить переобучение (train растёт, val — нет)
    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    epoch_val_loss = val_loss / total_val
    epoch_val_acc = correct_val / total_val

    print(f"Эпоха {epoch+1}/{epochs} | "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}% | "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%")

# Финальное предсказание на тесте
model.eval()
predictions = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())

submission = pd.DataFrame({"ImageId": range(1, len(predictions) + 1), "Label": predictions})
submission.to_csv("submission.csv", index=False)
print("submission.csv сохранён!")
