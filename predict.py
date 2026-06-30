"""Fill this in. That's the whole interface.

Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)
A hard 0 or 1 is fine if your method gives a yes/no answer.
"""

import sys
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.mobilenet import build_model

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def predict(image_path: str) -> float:
    model = build_model()
    model.load_state_dict(
        torch.load("weights/best_model.pth", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img)
        probs = F.softmax(logits, dim=1)
        screen_probability = probs[0, 1].item()

    return screen_probability

if __name__ == "__main__":
    score = predict(sys.argv[1])

    prediction = "SCREEN" if score >= 0.5 else "REAL"

    confidence = score if score >= 0.5 else (1 - score)

    print(f"Prediction : {prediction}")
    print(f"Confidence : {confidence*100:.2f}%")
    print(f"Fraud Score : {score:.4f}")