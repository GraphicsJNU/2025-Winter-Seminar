import torch

from config import device


def evaluate(model, test_loader, verbose=True):
    model.eval()

    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            pred = output.argmax(dim=1)
            correct += (pred == labels).sum().item()

    accuracy = correct / len(test_loader.dataset) * 100

    if verbose:
        print(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy
