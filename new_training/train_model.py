# new_training/train_heel_model.py
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from make_dataset import train_loader, val_loader  # re-use from previous file

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# -------------------------
# Model
# -------------------------
class TinyKeypointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # 256 -> 128 -> 64 -> 32 with pooling
        self.fc2 = nn.Linear(128, 2)             # (x_norm, y_norm)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 256 -> 128
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 128 -> 64
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)  # 64 -> 32
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # keep outputs in [0,1]
        return x

def main():
    model = TinyKeypointNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    NUM_EPOCHS = 20
    CHECKPOINT_PATH = Path("new_training/heel_model.pth")
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for imgs, targets, frame_idx in train_loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets, frame_idx in val_loader:
                imgs = imgs.to(DEVICE)
                targets = targets.to(DEVICE)
                preds = model(imgs)
                val_loss += criterion(preds, targets).item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}  train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved model to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()