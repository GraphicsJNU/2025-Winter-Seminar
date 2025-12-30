import torch
import torch.nn as nn

from config import device
from dataset import get_data_loaders
from evaluate import evaluate
from model import ResNet18
from train import train
from utils import set_seed


if __name__ == '__main__':
    print(f"Using device: {device}")
    set_seed(42)

    train_loader, test_loader = get_data_loaders()

    model = ResNet18().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-4)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_model_path = "best_model.pth"

    train(model, train_loader, test_loader, optimizer, loss_fn, save_path=best_model_path, log_dir="resnet_train")

    model.load_state_dict(torch.load(best_model_path))
    print("\n📦 Loaded best saved model for final evaluation:")

    evaluate(model, test_loader)
