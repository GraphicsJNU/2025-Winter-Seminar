import torch

from torchvision.utils import make_grid
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from config import device
from evaluate import evaluate


def train(
    model,
    train_loader,
    test_loader,
    optimizer,
    loss_fn,
    epochs=30,
    save_path="best_model.pth",
    log_dir="cnn_train"
):
    best_accuracy = 0.0
    count = 0
    patience = 12

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    writer = SummaryWriter(log_dir=f"runs/{log_dir}")

    for epoch in range(epochs):
        model.train()

        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("LearningRate", current_lr, epoch)

        print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

        acc = evaluate(model, test_loader, verbose=False)
        print(f"Test Accuracy: {acc:.2f}%")

        writer.add_scalar("Loss/train", total_loss, epoch)
        writer.add_scalar("Accuracy/test", acc, epoch)

        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved new best model with accuracy: {acc:.2f}%")
            count = 0
        else:
            count += 1

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)

        if epoch == 0:
            image_grid = make_grid(inputs[:16].cpu(), nrow=4, normalize=True)
            writer.add_image("Sample inputs", image_grid, epoch)

        if count > patience:
            print("⏹️ Early stopping triggered")
            break

    print("📁 TensorBoard logs saved in:", log_dir)
    writer.close()
