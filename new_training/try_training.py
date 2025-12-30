from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm
import albumentations as A

def pad_to_square(img, pts=None):
    """Pad image to square, adjust points if provided."""
    h, w = img.shape[:2]
    size = max(h, w)
    pad_top = (size - h) // 2
    pad_left = (size - w) // 2
    
    padded = cv2.copyMakeBorder(
        img, pad_top, size - h - pad_top, pad_left, size - w - pad_left,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    
    if pts is not None:
        pts = pts.copy()
        pts[:, 0] += pad_left
        pts[:, 1] += pad_top
        return padded, pts
    return padded
# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_gaussian_heatmaps(points_xy, visible, H, W, sigma=2.5):
    """
    points_xy: (K,2) in heatmap coords
    visible: (K,) bool
    """
    K = points_xy.shape[0]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = np.zeros((K, H, W), dtype=np.float32)

    for k in range(K):
        if not visible[k]:
            continue
        x, y = points_xy[k]
        out[k] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

    return out


# -------------------------
# Augmentation
# -------------------------
def get_augmentation(img_size):
    """Training augmentations — applied to image + keypoints together."""
    return A.Compose([
        A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, p=0.5),
        # A.HorizontalFlip(p=0.5),  # disable if asymmetric setup
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # A.GaussNoise(var_limit=(5, 25), p=0.2),
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def get_val_transform(img_size):
    """Validation/inference — just resize, no augmentation."""
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


# -------------------------
# Dataset
# -------------------------
class VideoKeypointDataset(Dataset):
    def __init__(
        self,
        video_path,
        points,
        visibility,
        frame_indices,
        img_size=256,
        hm_size=64,
        sigma=2.5,
        augment=True,
    ):
        self.video_path = str(video_path)
        self.points = points
        self.visibility = visibility
        self.frame_indices = np.array(frame_indices, dtype=int)

        self.img_size = img_size
        self.hm_size = hm_size
        self.sigma = sigma
        
        self.transform = get_augmentation(img_size) if augment else get_val_transform(img_size)
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        fi = int(self.frame_indices[idx])

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = self.cap.read()

        if not ok:
            raise RuntimeError(f"Failed to read frame {fi}")
            

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pts = self.points[fi].astype(np.float32)
        vis = self.visibility[fi] > 0.5

        frame_rgb, pts = pad_to_square(frame_rgb, pts)

        # Convert to list of tuples for albumentations
        keypoints = [tuple(pt) for pt in pts]

        # Apply augmentation (handles both image and keypoints)
        transformed = self.transform(image=frame_rgb, keypoints=keypoints)
        frame_aug = transformed['image']
        keypoints_aug = np.array(transformed['keypoints'], dtype=np.float32)

        # Check which keypoints are still in bounds after augmentation
        vis_aug = vis.copy()
        for k in range(len(keypoints_aug)):
            x, y = keypoints_aug[k]
            if x < 0 or x >= self.img_size or y < 0 or y >= self.img_size:
                vis_aug[k] = False

        # To tensor and normalize
        x = torch.from_numpy(frame_aug.astype(np.float32) / 255.0).permute(2, 0, 1)
        x = (x - self.mean) / self.std

        # Scale keypoints to heatmap coordinates
        pts_hm = keypoints_aug.copy()
        pts_hm[:, 0] *= self.hm_size / self.img_size
        pts_hm[:, 1] *= self.hm_size / self.img_size

        

        hms = make_gaussian_heatmaps(pts_hm, vis_aug, self.hm_size, self.hm_size, sigma=self.sigma)

        y = torch.from_numpy(hms)
        v = torch.from_numpy(vis_aug.astype(np.float32))

        return x, y, v


# -------------------------
# Model
# -------------------------
class ResNetPoseNet(nn.Module):
    """ResNet backbone with deconv upsampling (SimpleBaseline style)."""
    
    def __init__(self, K, hm_size=64, backbone="resnet18", pretrained=True):
        super().__init__()
        self.hm_size = hm_size

        if backbone == "resnet18":
            resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
            c_out = 512
        elif backbone == "resnet34":
            resnet = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT if pretrained else None)
            c_out = 512
        elif backbone == "resnet50":
            resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None)
            c_out = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Deconv head: 8 -> 16 -> 32 -> 64
        self.head = nn.Sequential(
            nn.ConvTranspose2d(c_out, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, K, kernel_size=1),
        )
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        f = self.backbone(x)
        h = self.head(f)
        return h


def masked_mse(pred, target, vis):
    vis = vis.unsqueeze(-1).unsqueeze(-1)
    num = ((pred - target) ** 2 * vis).sum()
    den = vis.sum() * pred.shape[-1] * pred.shape[-2] + 1e-8
    return num / den


# -------------------------
# Training function
# -------------------------
def train(
    labels_path,
    video_path,
    out_path="model.pt",
    img_size=512,
    hm_size=128,
    sigma=3.0,
    epochs=50,
    batch_size=32,
    lr_backbone=1e-4,
    lr_head=1e-3,
    seed=0,
    device=None,
):

        
    set_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    z = np.load(labels_path, allow_pickle=True)
    points = z["points"]
    visibility = z["visibility"]
    marker_names = list(z["marker_names"])
    K = len(marker_names)

    labeled_frames = np.where(visibility.sum(axis=1) > 0)[0]
    if len(labeled_frames) < 5:
        raise RuntimeError("Not enough labeled frames")

    ds = VideoKeypointDataset(
        video_path,
        points,
        visibility,
        labeled_frames,
        img_size=img_size,
        hm_size=hm_size,
        sigma=sigma,
        augment=False,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = ResNetPoseNet(K, hm_size=hm_size, backbone="resnet18", pretrained=True).to(device)
    
    # Differential learning rates
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": lr_backbone},
        {"params": model.head.parameters(), "lr": lr_head},
    ])
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"Training on {len(ds)} frames, {K} keypoints")
    print(f"LR: backbone={lr_backbone}, head={lr_head}")
    print(f"Sigma: {sigma}, Epochs: {epochs}")

    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        
        for x, y, v in dl:
            x, y, v = x.to(device), y.to(device), v.to(device)
            pred = model(x)
            loss = masked_mse(pred, y, v)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        epoch_loss = running / len(dl)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:03d} | loss {epoch_loss:.6f} | lr {current_lr:.2e}")
        
        scheduler.step()
        
        # Track best
        if epoch_loss < best_loss:
            best_loss = epoch_loss

    ckpt = {
        "state_dict": model.state_dict(),
        "marker_names": marker_names,
        "img_size": img_size,
        "hm_size": hm_size,
        "sigma": sigma,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"Saved model to: {out_path}")
    print(f"Best loss: {best_loss:.6f}")

    return model


if __name__ == "__main__":
    path_to_data = Path(r"D:\sfn\michael_wobble\recording_12_07_09_gmt-5__MDN_wobble_3")

    model = train(
        labels_path=path_to_data / "synchronized_videos" / "Camera_000_synchronized.labels.npz",
        video_path=path_to_data / "synchronized_videos" / "Camera_000_synchronized.mp4",
        out_path=path_to_data / "runs/v2_cam6_resnet18.pt",
        epochs=200,
        batch_size=16,
        lr_backbone=1e-4,
        lr_head=1e-3,
        sigma=2.5,
    )