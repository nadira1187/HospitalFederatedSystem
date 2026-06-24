import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import get_model


# ---------------------
# Transform (512 -> 224)
# ---------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ---------------------
# Dataset (CHANGE PATH PER CLIENT)
# ---------------------

train_path = "data/LungData/client3"

train_dataset = datasets.ImageFolder(
    root=train_path,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)


# ---------------------
# Model
# ---------------------

model = get_model()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# ---------------------
# Training
# ---------------------

def train():

    model.train()

    for epoch in range(2):

        for images, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()


# ---------------------
# Evaluation
# ---------------------

def test():

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in train_loader:

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    return correct / total


# ---------------------
# Flower Client
# ---------------------

class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in model.state_dict().values()]

    def set_parameters(self, parameters):
        params = zip(model.state_dict().keys(), parameters)

        state_dict = {
            k: torch.tensor(v)
            for k, v in params
        }

        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train()

        return self.get_parameters(config), len(train_dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        acc = test()

        print("Client3Accuracy:", acc)

        return 0.0, len(train_dataset), {"accuracy": acc}


fl.client.start_numpy_client(
    server_address="192.168.227.85:8080",
    client=FlowerClient(),
)