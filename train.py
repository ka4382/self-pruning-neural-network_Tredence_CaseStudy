import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from model import PruningNet
from utils import compute_sparsity_loss, calculate_sparsity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def train_model(lambda_val, epochs=10):
    model = PruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            cls_loss = criterion(outputs, labels)
            sparse_loss = compute_sparsity_loss(model)

            loss = cls_loss + lambda_val * sparse_loss

            loss.backward()
            optimizer.step()

        print(f"Lambda {lambda_val} | Epoch {epoch+1} done")

    acc = evaluate(model)
    sparsity = calculate_sparsity(model)

    print(f"Accuracy: {acc:.2f}% | Sparsity: {sparsity:.2f}%")
    return acc, sparsity


if __name__ == "__main__":
    lambda_values = [0.5, 1.0, 2.0]

    for lam in lambda_values:
        print(f"\nRunning for lambda = {lam}")
        train_model(lam)
