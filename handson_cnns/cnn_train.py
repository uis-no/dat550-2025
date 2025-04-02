import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layer (sees 28x28x1 image tensor)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Convolutional layer (sees 14x14x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer (sees 7x7x32 tensor)
        self.fc1 = nn.Linear(32 * 7 * 7, 120)
        # Fully connected layer (sees 120 tensor)
        self.fc2 = nn.Linear(120, 10) # Assuming 10 classes

    def forward(self, x):
        # Add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten image input
        x = x.view(-1, 32 * 7 * 7)
        # Add dropout layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
model = SimpleCNN()
print(model)

# Specify the loss and optimization method
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def count_parameters(model):
    # Implement a function to count the total number of trainable parameters in the model
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print(model_parameters)
    return sum([p.numel() for p in model_parameters])

# Assuming model is an instance of SimpleCNN
model = SimpleCNN()
total_params = count_parameters(model)
print(f"Total trainable parameters: {total_params}")

# Generate random dataset
num_samples = 10000  # Number of samples in the dataset
num_classes = 10    # Number of classes, for the output layer

# Images: Random tensors of shape (num_samples, 1, 28, 28)
images = torch.randn(num_samples, 1, 28, 28)
# Labels: Random integers representing classes for each sample
labels = torch.randint(0, num_classes, (num_samples,))

# Creating a dataset and dataloader
dataset = TensorDataset(images, labels)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SimpleCNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10  # Number of epochs to train for
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print average loss for the epoch
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}')