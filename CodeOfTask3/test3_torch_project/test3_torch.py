# test3_torch.py
# -*- coding: utf-8 -*-
"""
PyTorch rewrite of Paddle code (fundus disc/cup segmentation).
- Dataset, model (SeparableConv UNet), losses (Dice variants), train/val, inference
- No Paddle dependencies
"""

import os, cv2, random, argparse, math, re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------- Config -----------------------
images_file = '/home/yanggq/project/grading/Glaucoma_grading/training/multi-modality_images'  # fundus image root
gt_file     = '/home/yanggq/project/grading/task3_disc_cup_segmentation/training/Disc_Cup_Mask'  # mask root
val_ratio   = 0.1
image_size  = 512
BATCH_SIZE  = 8
iters       = 20000
optimizer_type = 'adam'
num_workers = 8
init_lr     = 1e-3
save_dir    = "trained_models_torch"

# ------------------- Simple paired transforms -------------------
def center_crop(img, size_hw):
    h, w = img.shape[:2]
    th, tw = size_hw
    th = min(th, h); tw = min(tw, w)
    i = (h - th)//2; j = (w - tw)//2
    return img[i:i+th, j:j+tw]

def resize_image(img, size_hw, is_mask=False):
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(img, (size_hw[1], size_hw[0]), interpolation=interp)

def random_hflip_pair(img, mask, p=0.5):
    if random.random() < p:
        img  = img[:, ::-1].copy()
        mask = mask[:, ::-1].copy()
    return img, mask

def random_vflip_pair(img, mask, p=0.5):
    if random.random() < p:
        img  = img[::-1, :].copy()
        mask = mask[::-1, :].copy()
    return img, mask

def random_rotate_pair(img, mask, deg=60):
    angle = random.uniform(-deg, deg)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img2  = cv2.warpAffine(img,  M, (w, h), flags=cv2.INTER_LINEAR,  borderMode=cv2.BORDER_REFLECT_101)
    mask2 = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return img2, mask2

# ----------------------- Dataset -----------------------
class FundusDataset(Dataset):
    """
    images_file/ID/ID.jpg
    gt_file/ID.png  (values: 0=cup, 128=disc, 255=background)
    Returns (for train/val):
      img: float32 [3,H,W] in [0,1]
      mask: int64  [H,W] with classes {0,1,2}
    For test:
      (img, id, orig_h, orig_w)
    """
    def __init__(self, image_file, gt_path=None, filelists=None, mode='train',
                 do_aug=True, img_size=512):
        super().__init__()
        self.image_file = image_file
        self.gt_path = gt_path
        self.mode = mode.lower()
        self.do_aug = do_aug and (self.mode=='train')
        self.img_size = img_size

        ids = sorted(os.listdir(self.image_file))
        if filelists is not None:
            keep = set(filelists)
            ids = [i for i in ids if i in keep]
        self.ids = ids

    def __len__(self): return len(self.ids)

    def _read_img(self, idx):
        p = os.path.join(self.image_file, idx, f"{idx}.jpg")
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None: raise FileNotFoundError(p)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def _read_mask(self, idx):
        p = os.path.join(self.gt_path, f"{idx}.png")
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None: raise FileNotFoundError(p)
        # Map values: 0->0 (cup), 128->1 (disc), 255->2 (bg)
        mask = np.zeros_like(m, dtype=np.uint8)
        mask[m==0]   = 0
        mask[m==128] = 1
        mask[m==255] = 2
        return mask

    def __getitem__(self, i):
        idx = self.ids[i]
        img = self._read_img(idx)
        h0, w0 = img.shape[:2]

        # center crop to square, then resize
        s = min(h0, w0)
        img = center_crop(img, (s, s))
        img = resize_image(img, (self.img_size, self.img_size), is_mask=False)

        if self.mode == 'test':
            img_t = torch.from_numpy(img.astype(np.float32).transpose(2,0,1) / 255.0)
            return img_t, idx, h0, w0

        mask = self._read_mask(idx)
        mask = center_crop(mask, (s, s))
        mask = resize_image(mask, (self.img_size, self.img_size), is_mask=True)

        if self.do_aug:
            img, mask = random_hflip_pair(img, mask, 0.5)
            img, mask = random_vflip_pair(img, mask, 0.5)
            img, mask = random_rotate_pair(img, mask, 60)

        img_t  = torch.from_numpy(img.astype(np.float32).transpose(2,0,1)/255.0)
        mask_t = torch.from_numpy(mask.astype(np.int64))

        if self.mode == 'eval':
            return img_t, mask_t, idx
    
        return img_t, mask_t

# ----------------------- Model -----------------------
class SeparableConv2D(nn.Module):
    """Depthwise separable conv: depthwise + pointwise"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, dilation,
                                   groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# ---- Encoder ----
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu1 = nn.ReLU(inplace=False)   # 改：False
        self.sep1  = SeparableConv2D(in_channels, out_channels, 3, 1, 1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)   # 改：False
        self.sep2  = SeparableConv2D(out_channels, out_channels, 3, 1, 1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.pool  = nn.MaxPool2d(3, stride=2, padding=1)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        res = self.residual(x)
        y = self.relu1(x)          # 非就地
        y = self.sep1(y); y = self.bn1(y)
        y = self.relu2(y)          # 非就地
        y = self.sep2(y); y = self.bn2(y)
        y = self.pool(y)
        return y + res             # 非就地加法

# ---- Decoder ----
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu1   = nn.ReLU(inplace=False)  # 改：False
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1     = nn.BatchNorm2d(out_channels)
        self.relu2   = nn.ReLU(inplace=False)  # 改：False
        self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2     = nn.BatchNorm2d(out_channels)
        self.ups     = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False)
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        res = self.ups(self.res_conv(x))
        y = self.relu1(x)          # 非就地
        y = self.deconv1(y); y = self.bn1(y)
        y = self.relu2(y)          # 非就地
        y = self.deconv2(y); y = self.bn2(y)
        y = self.ups(y)
        return y + res             # 非就地加法

class CupDiscUNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu  = nn.ReLU(inplace=False)
        # encoders
        in_channels = 32
        self.encoders = nn.ModuleList()
        for out_ch in [64,128,256]:
            self.encoders.append(Encoder(in_channels, out_ch))
            in_channels = out_ch
        # decoders
        self.decoders = nn.ModuleList()
        for out_ch in [256,128,64,32]:
            self.decoders.append(Decoder(in_channels, out_ch))
            in_channels = out_ch
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        for enc in self.encoders:
            y = enc(y)
        for dec in self.decoders:
            y = dec(y)
        y = self.head(y)
        return y  # logits [N,C,H,W]

# ----------------------- Losses -----------------------
def one_hot(labels, num_classes):
    # labels: [N,H,W] int64
    n, h, w = labels.size(0), labels.size(1), labels.size(2)
    y = torch.zeros((n, num_classes, h, w), device=labels.device, dtype=torch.float32)
    return y.scatter_(1, labels.unsqueeze(1), 1.0)

class DiceMetric(nn.Module):
    """As metric (softmax->prob), returns mean Dice over classes"""
    def __init__(self, eps=1e-5, ignore_index=None):
        super().__init__()
        self.eps = eps
        self.ignore_index = ignore_index
    def forward(self, logits, labels):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        if self.ignore_index is not None:
            mask = (labels != self.ignore_index).float()
            probs = probs * mask.unsqueeze(1)
        labels_1h = one_hot(labels.clamp_min(0), num_classes)
        inter = torch.sum(probs * labels_1h, dim=(0,2,3))
        card  = torch.sum(probs + labels_1h, dim=(0,2,3))
        dice = (2*inter / (card + self.eps)).mean()
        return dice

class MinusDiceLoss(nn.Module):
    """1 - Dice(probs, onehot)"""
    def __init__(self, eps=1e-5, ignore_index=None):
        super().__init__()
        self.metric = DiceMetric(eps, ignore_index)
    def forward(self, logits, labels):
        return 1.0 - self.metric(logits, labels)

class DiceFromArgmax(nn.Module):
    """Replicates your MinusDiceLoss2/DiceLoss2 behavior using argmax masks.
       Weights: cup 0.30, disc 0.42, bg 0.28 (as in your code)."""
    def __init__(self, weights=(0.30, 0.42, 0.28), eps=1e-5):
        super().__init__()
        self.w = torch.tensor(weights, dtype=torch.float32)
        self.eps = eps
    def forward(self, logits, labels):
        # logits -> hard mask
        pred = torch.argmax(logits, dim=1)  # [N,H,W]
        num_classes = logits.shape[1]
        # one-hot both
        pred_1h = one_hot(pred, num_classes).float()
        labels_1h = one_hot(labels, num_classes).float()
        inter = torch.sum(pred_1h * labels_1h, dim=(0,2,3))
        card  = torch.sum(pred_1h + labels_1h, dim=(0,2,3))
        dice_per_class = (2*inter / (card + self.eps))
        # pad weights if needed
        w = self.w.to(dice_per_class.device)
        if w.numel() != dice_per_class.numel():
            w = torch.ones_like(dice_per_class) / dice_per_class.numel()
        dice = torch.sum(w * dice_per_class)
        return 1.0 - dice  # as loss

# ----------------------- Train / Val -----------------------
@torch.no_grad()
def evaluate(model, loader, criterion_loss, criterion_metric, device):
    model.eval()
    losses, dices = [], []
    for img, mask in loader:
        img = img.to(device); mask = mask.to(device)
        logits = model(img)
        loss = criterion_loss(logits, mask)
        metric = 1.0 - DiceFromArgmax()(logits, mask) if isinstance(criterion_metric, DiceFromArgmax) else criterion_metric(logits, mask)
        losses.append(loss.item())
        dices.append(metric.item())
    return float(np.mean(losses)), float(np.mean(dices))

def train_loop(args):
    # split
    filelists = sorted(os.listdir(images_file))
    #train_ids, val_ids = train_test_split(filelists, test_size=val_ratio, random_state=42)
    # 最后 20 个作为测试集
    val_ids = filelists[-20:]
    # 其余的作为训练集
    train_ids = filelists[:-20]

    print(f"Total Nums: {len(filelists)}, train: {len(train_ids)}, val: {len(val_ids)}")
    print("val sample:", val_ids)

    train_ds = FundusDataset(images_file, gt_file, filelists=train_ids, mode='train', do_aug=True, img_size=image_size)
    val_ds   = FundusDataset(images_file, gt_file, filelists=val_ids,   mode='val',   do_aug=False, img_size=image_size)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = CupDiscUNet(num_classes=3).to(device)

    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=1e-4)

    # You used CrossEntropyLoss(axis=1) + Dice metric; keep that.
    criterion_loss = nn.CrossEntropyLoss()
    # Your metric DiceLoss2 approximates dice on argmax with class weights -> use DiceFromArgmax
    criterion_metric = DiceFromArgmax(weights=(0.30,0.42,0.28))

    best_dice = -1e9
    os.makedirs(save_dir, exist_ok=True)

    it = 0
    model.train()
    avg_loss_buf, avg_dice_buf = [], []
    while it < iters:
        for img, mask in train_loader:
            it += 1
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            logits = model(img)
            loss = criterion_loss(logits, mask)
            metric_val = 1.0 - DiceFromArgmax()(logits, mask)  # convert to "dice" for logging

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            avg_loss_buf.append(loss.item())
            avg_dice_buf.append(metric_val.item())

            if it % 32 == 0:
                avg_loss = float(np.mean(avg_loss_buf)); avg_loss_buf.clear()
                avg_dice = float(np.mean(avg_dice_buf)); avg_dice_buf.clear()
                print(f"[TRAIN] iter={it}/{iters} avg_loss={avg_loss:.4f} avg_dice={avg_dice:.4f}")

            if it % 160 == 0:
                vloss, vdice = evaluate(model, val_loader, criterion_loss, criterion_metric, device)
                print(f"[EVAL ] iter={it}/{iters} avg_loss={vloss:.4f} dice={vdice:.4f}")
                if vdice >= best_dice:
                    best_dice = vdice
                    out_dir = os.path.join(save_dir, f"best_model_{best_dice:.4f}")
                    os.makedirs(out_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
                    torch.save(optimizer.state_dict(), os.path.join(out_dir, "optimizer.pt"))
                    print("[SAVE ]", out_dir)
                model.train()

            if it >= iters:
                break

# ----------------------- Visualization eval (single batch) -----------------------
@torch.no_grad()
def val_e(weights_path, val_ids=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CupDiscUNet(num_classes=3).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    if val_ids is None:
        val_ids = ['0016']

    ds = FundusDataset(images_file, gt_file, filelists=val_ids, mode='val', do_aug=False, img_size=image_size)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    import matplotlib.pyplot as plt
    for img, mask in dl:
        img = img.to(device)
        logits = model(img)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.float32)
        gt   = mask[0].cpu().numpy().astype(np.float32)

        plt.figure(figsize=(10,4))
        plt.subplot(1,3,1); plt.title("GT");   plt.imshow(gt, cmap='gray')
        plt.subplot(1,3,2); plt.title("Pred"); plt.imshow(pred, cmap='gray')
        # simple overlay
        rgb = (img[0].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
        overlay = rgb.copy()
        overlay[pred==1] = (0,255,0)
        overlay[pred==2] = (255,0,0)
        plt.subplot(1,3,3); plt.title("Overlay"); plt.imshow(overlay); plt.tight_layout(); plt.show()

# ----------------------- Inference (save BMP masks) -----------------------
@torch.no_grad()
def infer_to_bmp(weights_path, test_path='val_data/multi-modality_images', out_dir='Disc_Cup_Segmentations'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CupDiscUNet(num_classes=3).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ds = FundusDataset(test_path, gt_path=None, filelists=None, mode='test', do_aug=False, img_size=image_size)
    os.makedirs(out_dir, exist_ok=True)

    ids = sorted(os.listdir(test_path))
    print(ids)
    for img, idx, h, w in ds:
        img = img.unsqueeze(0).to(device)
        logits = model(img)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.float32)
        # map back to 0/128/255 palette
        pred[pred==1] = 128
        pred[pred==2] = 255
        # resize back to square min(h,w), then pad to original like original code
        min_side = min(h, w)
        pred_sq = cv2.resize(pred, (min_side, min_side), interpolation=cv2.INTER_NEAREST)
        padding = (max(h,w) - min(h,w))//2
        if h >= w:
            # pad left/right
            pad_l = padding; pad_r = max(h,w) - min(h,w) - pad_l
            pred_full = cv2.copyMakeBorder(pred_sq, 0, 0, pad_r, pad_l, cv2.BORDER_CONSTANT, value=255)
        else:
            # pad top/bottom
            pad_t = padding; pad_b = max(h,w) - min(h,w) - pad_t
            pred_full = cv2.copyMakeBorder(pred_sq, pad_b, pad_t, 0, 0, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(os.path.join(out_dir, f"{idx}.bmp"), pred_full)

@torch.no_grad()
def eval_bmp(weights_path, test_path='val_data/multi-modality_images', out_dir='Disc_Cup_Segmentations'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CupDiscUNet(num_classes=3).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ds = FundusDataset(test_path, gt_path=None, filelists=None, mode='test', do_aug=False, img_size=image_size)
    os.makedirs(out_dir, exist_ok=True)

    ids = sorted(os.listdir(test_path))
    print(ids)
    for img, mask, idx in ds:
        

# ----------------------- Main -----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="train", choices=["train","val_e","infer","eval"])
    ap.add_argument("--weights", type=str, default="trained_models_torch/best_model_0.9000/model.pt")
    args = ap.parse_args()

    if args.mode == "train":
        train_loop(args)
    elif args.mode == "val_e":
        val_e(args.weights, val_ids=['0016'])
    elif args.mode == "eval":
        val_e(args.weights, val_ids=['0016'])
    else:
        infer_to_bmp(args.weights, test_path='val_data/multi-modality_images', out_dir='Disc_Cup_Segmentations')
