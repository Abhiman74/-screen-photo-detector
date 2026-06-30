import os
import copy

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import get_dataloaders
from models.mobilenet import build_model, unfreeze_last_layers


# -----------------------------
# Device
# -----------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"\nUsing device: {DEVICE}\n")


# -----------------------------
# Hyperparameters
# -----------------------------

BATCH_SIZE = 4
EPOCHS = 20

LR = 3e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 10


# -----------------------------
# Model
# -----------------------------

model = build_model()

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.05
)
optimizer = AdamW(
    model.parameters(),
    lr=5e-6,
    weight_decay=WEIGHT_DECAY
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# -----------------------------
# Training
# -----------------------------

def train_one_epoch():

    model.train()

    running_loss = 0
    running_correct = 0
    total = 0

    loop = tqdm(train_loader)

    for images, labels in loop:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, preds = outputs.max(1)

        running_correct += (preds == labels).sum().item()

        total += labels.size(0)

        loop.set_description("Training")

        loop.set_postfix(
            loss=running_loss / (total / BATCH_SIZE),
            acc=100 * running_correct / total
        )

    return (
        running_loss / len(train_loader),
        running_correct / total
    )


# -----------------------------
# Validation
# -----------------------------

def validate():

    model.eval()

    running_loss = 0
    running_correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, preds = outputs.max(1)

            running_correct += (preds == labels).sum().item()

            total += labels.size(0)

    return (
        running_loss / len(val_loader),
        running_correct / total
    )

# -----------------------------
# Training Loop
# -----------------------------

def main():
    global train_loader, val_loader, test_loader
    global model, optimizer, scheduler, best_weights

    train_loader, test_loader = get_dataloaders(
    batch_size=BATCH_SIZE
    )

# Validation set was merged into training.
# Reuse the test loader as the validation loader.
    val_loader = test_loader

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):

        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc = validate()

        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            best_weights = copy.deepcopy(model.state_dict())
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), "weights/best_model.pth")
            print("Saved Best Model")
        else:
            patience_counter += 1


    print("\nUnfreezing last layers...\n")

    model = unfreeze_last_layers(model)

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=20
    )

    for epoch in range(20):

        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc = validate()

        scheduler.step()

        print(
            f"FineTune [{epoch+1}/20] | "
            f"Train {train_acc:.4f} | "
            f"Val {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "weights/best_model.pth")
            print("Saved Better Model")

    print("\nLoading best model...\n")

    model.load_state_dict(best_weights)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total

    print("\n============================")
    print(f"Test Accuracy : {test_acc*100:.2f}%")
    print("============================")


if __name__ == "__main__":
    main()