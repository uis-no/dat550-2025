import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Subset






# 📦 Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 🔁 Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),              # Match model input size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),        # Mean for R, G, B
                         (0.5, 0.5, 0.5))        # Std for R, G, B
])

# 📊 Subset of CIFAR-10 to avoid overload
trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
testset_full = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

train_subset = Subset(trainset_full, range(1000))  # Smaller for SLURM/GPU limits
test_subset = Subset(testset_full, range(200))

trainloader = torch.utils.data.DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(test_subset, batch_size=8, shuffle=False, num_workers=0)

# 🧠 Load ResNet-18 without crashing
try:
    print("Loading pretrained ResNet-18...")
    resnet = models.resnet18(pretrained=True)
except Exception as e:
    print(f"⚠️ Could not load pretrained weights: {e}")
    resnet = models.resnet18(pretrained=False)

# ✂️ Replace the classifier head
resnet.fc = nn.Linear(resnet.fc.in_features, 10)
resnet = resnet.to(device)

# 🎯 Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# 🔁 Training loop
for epoch in range(3):  # Keep it fast
    resnet.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader):.4f}")

# 📈 Evaluation
resnet.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy on 200 samples: {100 * correct / total:.2f}%")