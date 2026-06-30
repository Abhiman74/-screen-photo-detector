import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)
def build_model():
    model = mobilenet_v3_large(
        weights=MobileNet_V3_Large_Weights.DEFAULT
    )

    # Freeze backbone initially (train.py will unfreeze it later for fine-tuning)
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[0].in_features
    print(f"Classifier input features: {in_features}")

    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 2)
    )

    return model


def unfreeze_last_layers(model):
    for param in model.features[-3:].parameters():
        param.requires_grad = True

    return model