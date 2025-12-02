import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt


def main():
    # ==========================
    # 🔧 CONFIG
    # ==========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU mode")

    train_dir = "C:/Users/zalut/PycharmProjects/TomatoGPU_ViT/tomato_dataset/train"
    val_dir = "C:/Users/zalut/PycharmProjects/TomatoGPU_ViT/tomato_dataset/valid"
    num_classes = 11      # змінити, якщо інша кількість класів
    batch_size = 16
    epochs = 10
    lr = 1e-4
    image_size = 224

    # ==========================
    # 🧠 DATASET
    # ==========================
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)  # 👈 тут 0 для Windows
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Classes: {train_data.classes}")

    # ==========================
    # 🧩 MODEL (Vision Transformer)
    # ==========================
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # ==========================
    # 📈 TRAINING LOOP
    # ==========================
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        preds, targets = [], []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            targets.extend(labels.cpu().numpy())

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='weighted')
        train_losses.append(train_loss / len(train_loader))

        # ==== VALIDATION ====
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Acc: {acc:.4f}, F1: {f1:.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

    # ==========================
    # 📊 VISUALIZATION
    # ==========================
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(val_accs, label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.show()

    # ==========================
    # 💾 SAVE MODEL
    # ==========================
    torch.save(model.state_dict(), "vit_tomato_model.pth")
    print("Model saved as vit_tomato_model.pth ✅")


if __name__ == "__main__":
    main()

