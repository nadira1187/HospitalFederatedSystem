import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import Net

# Load MNIST
transform = transforms.ToTensor()

trainset = datasets.MNIST(
    root="./data",
    train=True,
    download=False,
    transform=transform,
)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

# Model
model = Net()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train for 1 epoch
for epoch in range(1):
    running_loss = 0.0

    for images, labels in trainloader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

print("Training completed!")