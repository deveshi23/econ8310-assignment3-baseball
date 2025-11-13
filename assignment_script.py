import os
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


# IDX Data reader
# Reads as IDX file and returns a numpy array
def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        if magic == 2051:  # images
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        elif magic == 2049:  # labels
            num_labels = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f"Invalid magic number {magic} in {filename}")
    return data


# Fashion MNIST Dataset
class FashionMNISTCustom(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if train:
            images_path = os.path.join(root, 'train-images-idx3-ubyte.gz')
            labels_path = os.path.join(root, 'train-labels-idx1-ubyte.gz')
        else:
            images_path = os.path.join(root, 't10k-images-idx3-ubyte.gz')
            labels_path = os.path.join(root, 't10k-labels-idx1-ubyte.gz')

        self.images = read_idx(images_path)
        self.labels = read_idx(labels_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if index >= len(self.labels):
            raise IndexError("Index out of range for labels")

        img = self.images[index]
        label = int(self.labels[index])  # convert before tensor
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        label = torch.tensor(label, dtype=torch.long)
        return img, label



# Neural Network Model
class SimpleFashionMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

# Training function
def train_model(model, train_loader, epochs=5, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

    # Correct save method
    torch.save(model.state_dict(), "fashion_mnist_model.pth")
    print("The model has been saved as fashion_mnist_model.pth")


# Main training
if __name__ == "__main__":
    data_root = "data" # Directory which contains the FashionMNIST gz files
    train_dataset = FashionMNISTCustom(root=data_root, train=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = SimpleFashionMNISTNet()
    train_model(model, train_loader, epochs=5)
    torch.save(model.state_dict(), "fashion_mnist_model.pth")


# Evaluation model
def evaluate_model(model_path, test_data_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleFashionMNISTNet()
    model.load_state_dict(torch.load("fashion_mnist_model.pth"))
    model.eval()


    test_dataset = FashionMNISTCustom(root=test_data_root, train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on the test set: {100 * correct / total:.2f}%")

if __name__=="__main__":
    evaluate_model("fashion_mnist_model.pth", "data")


