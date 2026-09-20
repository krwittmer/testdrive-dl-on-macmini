"""
Baseline CNN "Hello, world" code impl. w/ MNIST image data set.

CNN architecture needs to support MNIST images are 1 x 28 x 28. 
After the 3 conv/pool blocks above, the tensor is 128 x 1 x 1.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix


class MNISTCnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train(model, compute_device, train_loader, optimizer, loss_fn):
    model.train()

    total_loss = 0

    for images, labels in train_loader:
        images = images.to(compute_device)
        labels = labels.to(compute_device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def test(model, compute_device, test_loader, loss_fn):
    model.eval()

    total_loss = 0
    correct = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(compute_device)
            labels = labels.to(compute_device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)

    print(f"Test loss: {avg_loss:.4f}")
    print(f"Test accuracy: {accuracy:.2%}")

    print("\nPrecision, Recall, and F1-score:")
    print(classification_report(
        all_labels,
        all_predictions,
        digits=4
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))


def plot_losses(train_losses):
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("CNN Training Loss on MNIST")
    plt.grid(True)
    plt.savefig("mnist_cnn_loss.png")
    plt.show()


def main():
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {compute_device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False
    )

    model = MNISTCnn().to(compute_device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []

    EPOCHS = 15  # More reasonable for MNIST
    PATIENCE = 5  # For early stopping

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        
        train_loss = train(model, compute_device, train_loader, optimizer, loss_fn)
        train_losses.append(train_loss)
        
        print(f"Training loss: {train_loss:.4f}")
        
        # Only evaluate on test set at end, use validation during training
        if epoch % 5 == 0 or epoch == EPOCHS:
            test(model, compute_device, test_loader, loss_fn)

    plot_losses(train_losses)


if __name__ == "__main__":
    main()

