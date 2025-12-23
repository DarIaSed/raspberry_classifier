import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

train_dir = "C:\\Users\\dasha\\raspberry_classifier\\data\\datasets\\train"
val_dir = "C:\\Users\\dasha\\raspberry_classifier\\data\\datasets\\val"

img_size = 224
learning_rate = 0.001
num_epochs = 10
batch_size = 32

if os.path.exists(train_dir):
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print("Классы:", class_names)
    num_classes = len(class_names)

class_counts = {}
total_images = 0

for class_name in class_names:
    class_path = os.path.join(train_dir, class_name)
    if os.path.exists(class_path):
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[class_name] = len(images)
        total_images += len(images)

print(f"\nВсего изображений в train: {total_images}")
print("\nРаспределение по классам:")
for class_name, count in class_counts.items():
    print(f"  {class_name}: {count} изображений")

plt.figure(figsize=(10, 5))
bars = plt.bar(class_counts.keys(), class_counts.values())
plt.title('Распределение изображений по классам малины')
plt.xlabel('Классы малины')
plt.ylabel('Количество изображений')
plt.xticks(rotation=45, ha='right')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()

train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

print(f"\nКлассов: {num_classes}")
print(f"Картинок для обучения: {len(train_dataset)}")
print(f"Картинок для проверки: {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=0
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=0
)


class RaspberryClassifierResNet(nn.Module):
    def __init__(self, num_classes):
        super(RaspberryClassifierResNet, self).__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }


raspberry_model = RaspberryClassifierResNet(num_classes=num_classes)
results = train_model(
    raspberry_model, 
    train_loader, 
    val_loader, 
    num_epochs, 
    learning_rate, 
    device
)


torch.save(results['model'].state_dict(), 'raspberry_model.pth')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(results['train_losses'], label='Train Loss', color='blue', linestyle='-')
plt.plot(results['val_losses'], label='Val Loss', color='red', linestyle='--')
plt.title('Потери при обучении')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(results['train_accuracies'], label='Train Accuracy', color='blue', linestyle='-')
plt.plot(results['val_accuracies'], label='Val Accuracy', color='red', linestyle='--')
plt.title('Точность при обучении')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_results.png')
print("Графики обучения сохранены как 'training_results.png'")
# Сохранение метрик для DVC
metrics = {
    'train_accuracy': results['train_accuracies'][-1],
    'val_accuracy': results['val_accuracies'][-1],
    'train_loss': results['train_losses'][-1],
    'val_loss': results['val_losses'][-1]
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"Метрики сохранены в metrics.json: {metrics}")
plt.show()