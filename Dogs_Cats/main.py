import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Пути к данным (на Kaggle: /kaggle/input/dogs-vs-cats-redux-kernels-edition/...)
image_dir = "train"
test_dir = "test1"

all_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
print(f"Train: {len(train_files)} | Val: {len(val_files)}")


class CatDogDataset(Dataset):
    def __init__(self, filenames, directory, is_test=False, transform=None):
        self.filenames = filenames
        self.dir = directory
        self.is_test = is_test
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        image = Image.open(os.path.join(self.dir, name)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if not self.is_test:
            # Метка берётся прямо из имени файла: dog.123.jpg -> собака
            label = 1 if "dog" in name.lower() else 0
            return image, torch.tensor(label, dtype=torch.long)
        return image


# На обучении добавляем случайное отражение — простая аугментация,
# которая слегка увеличивает разнообразие данных и снижает переобучение
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_loader = DataLoader(
    CatDogDataset(train_files, image_dir, transform=train_transform),
    batch_size=32, shuffle=True,
)
val_loader = DataLoader(
    CatDogDataset(val_files, image_dir, transform=val_transform),
    batch_size=32, shuffle=False,
)

if os.path.exists(test_dir):
    test_files = sorted(f for f in os.listdir(test_dir) if f.endswith(".jpg"))
    test_loader = DataLoader(
        CatDogDataset(test_files, test_dir, is_test=True, transform=val_transform),
        batch_size=32, shuffle=False,
    )
else:
    test_files, test_loader = [], None


class CatsDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Три свёрточных блока: 128×128 -> 16×16 с 64 каналами
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Linear(64 * 16 * 16, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


device = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
model = CatsDogCNN().to(device)
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
        correct_train += (outputs.argmax(1) == labels).sum().item()
        total_train += labels.size(0)

    train_loss /= total_train
    train_accuracy = correct_train / total_train

    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            correct_val += (outputs.argmax(1) == labels).sum().item()
            total_val += labels.size(0)

    val_loss /= total_val
    val_accuracy = correct_val / total_val

    print(f"Эпоха [{epoch+1}/{epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy*100:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy*100:.2f}%")

# Финальное предсказание
if test_loader is not None and len(test_files) > 0:
    print("Генерация финального сабмита...")
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            predictions.extend(model(images).argmax(1).cpu().numpy())

    # id берём из имени файла: 123.jpg -> 123
    image_ids = [int(f.split(".")[0]) for f in test_files]
    submission = pd.DataFrame({"id": image_ids, "label": predictions})
    submission.to_csv("submission_cats_dogs.csv", index=False)
    print("submission_cats_dogs.csv сохранён!")
else:
    print("Папка test1/ пуста или не найдена. Обучение завершено без сабмита.")
