import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import VGG16_Weights
from torch.utils.data import Subset

# Device setup (safe for SLURM)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 2. Load CIFAR-10 but only a subset
trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
testset_full = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

# Use smaller subsets for training/testing (e.g., 1000 train, 200 test)
train_subset = Subset(trainset_full, range(1000))
test_subset = Subset(testset_full, range(200))

trainloader = torch.utils.data.DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(test_subset, batch_size=8, shuffle=False, num_workers=0)

# 3. Load VGG-16 safely
try:
    print("Trying to load pretrained VGG-16...")
    vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
except Exception as e:
    print(f"⚠️ Could not load pretrained weights: {e}")
    print("Falling back to untrained VGG-16.")
    vgg16 = models.vgg16(weights=None)

# Freeze features to reduce training time/memory
for param in vgg16.features.parameters():
    param.requires_grad = False

# Modify classifier for CIFAR-10
vgg16.classifier[6] = nn.Linear(4096, 10)
vgg16 = vgg16.to(device)

# 4. Set up training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16.classifier.parameters(), lr=0.001)

# 5. Training loop
for epoch in range(3):  # fewer epochs for quick test
    vgg16.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = vgg16(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader):.4f}")

# 6. Evaluation
vgg16.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = vgg16(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy on 200 samples: {100 * correct / total:.2f}%")