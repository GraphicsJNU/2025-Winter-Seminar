import sys
import torch
import numpy as np

from torchvision import transforms
from PIL import Image

from model import ResNet18
from config import device


def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    np_img = np.array(image)

    corners = [np_img[0, 0], np_img[0, -1], np_img[-1, 0], np_img[-1, -1]]
    background_mean = np.mean(corners)

    if np.mean(np_img) < background_mean:
        np_img = 255 - np_img

    threshold = (np.max(np_img.astype(np.uint16)) + np.min(np_img.astype(np.uint16))) / 2
    np_img = np.where(np_img > threshold, 255, 0).astype(np.uint8)

    debug = Image.fromarray(np_img)
    debug.resize((280, 280), Image.Resampling.NEAREST).save("debug_transformed_input.png")

    tensor = transforms.ToTensor()(debug)
    tensor = transforms.Normalize((0.1307,), (0.3081,))(tensor)

    return tensor.unsqueeze(0)


def predict(model, image_tensor):
    model.eval()

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    return predicted_class, confidence


if __name__ == '__main__':
    if len(sys.argv) != 2:
        assert "Usage python image_test.py path_to_image.png"
        sys.exit(1)

    image_path = sys.argv[1]

    model = ResNet18().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))

    image_tensor = preprocess_image(image_path)
    digit, conf = predict(model, image_tensor)

    print(f"🧠 Predicted Digit: {digit} (Confidence: {conf * 100:.2f}%)")
