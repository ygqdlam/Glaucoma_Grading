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

    	
@torch.no_grad()
def eval_to_bmp(weights_path, test_path='val_data/multi-modality_images', gt_dir='', out_dir='Disc_Cup_Segmentations'):
    image_size  = 512

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CupDiscUNet(num_classes=3).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "Pred"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "GT"), exist_ok=True)

    filelists = sorted(os.listdir(test_path))
    #train_ids, val_ids = train_test_split(filelists, test_size=val_ratio, random_state=42)
    # 最后 20 个作为测试集
    val_ids = filelists[-20:]
    # 其余的作为训练集
    train_ids = filelists[:-20]

    print(f"Total Nums: {len(filelists)}, train: {len(train_ids)}, val: {len(val_ids)}")
    print("val sample:", val_ids)

    ds = FundusDataset(test_path, gt_dir, filelists=val_ids, mode='eval', do_aug=True, img_size=image_size)

    for img, mask, idx in ds:
        img = img.unsqueeze(0).to(device)
        logits = model(img)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        # map back to 0/128/255 palette
        pred[pred == 1] = 128
        pred[pred == 2] = 255


        # map back to 0/128/255 palette
        mask[mask == 1] = 128
        mask[mask == 2] = 255
        mask = mask.cpu().numpy().astype(np.uint8)

        # 先缩放到原图的尺寸 (w,h)
        #pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        #pred_resized = cv2.resize(pred, interpolation=cv2.INTER_NEAREST)
        #mask_resized = cv2.resize(mask, interpolation=cv2.INTER_NEAREST)
        # 保存为 PNG
        cv2.imwrite(os.path.join(out_dir, "Pred", f"{idx}.png"), pred)
        cv2.imwrite(os.path.join(out_dir, "GT", f"{idx}.png"), mask)

    per_image_df, micro, metrics  = eval_dir(os.path.join(out_dir, "GT"), os.path.join(out_dir, "Pred"), classes=(0,1,2), save_per_image_csv="per_image_metrics.csv")
    return per_image_df, micro, metrics
    	
import cv2
import numpy as np
from typing import Tuple, Optional

# ---------- 工具函数 ----------
def to_class_index(mask: np.ndarray) -> np.ndarray:
    """将 0/128/255 调色板映射为 0/1/2；若已是 0/1/2 直接返回"""
    uniq = set(np.unique(mask).tolist())
    if uniq <= {0, 1, 2}:
        return mask.astype(np.uint8)
    out = np.zeros_like(mask, dtype=np.uint8)
    out[mask == 0]   = 0  # cup
    out[mask == 128] = 1  # disc
    out[mask == 255] = 2  # background
    return out

def largest_component(bin_mask: np.ndarray) -> np.ndarray:
    """只保留最大连通域，去除小噪点"""
    bin_mask = (bin_mask > 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(bin_mask)
    if num <= 1:
        return bin_mask
    max_area = 0
    max_label = 0
    for lb in range(1, num):
        area = (labels == lb).sum()
        if area > max_area:
            max_area = area
            max_label = lb
    return (labels == max_label).astype(np.uint8)

def diameters_from_mask(bin_mask: np.ndarray, method: str = "ellipse") -> Tuple[float, float, int]:
    """
    从二值 mask 估计垂直/水平直径
    返回: (vertical_diameter, horizontal_diameter, area_in_pixels)
    """
    bin_mask = (bin_mask > 0).astype(np.uint8)
    area = int(bin_mask.sum())
    if area == 0:
        return 0.0, 0.0, 0

    # 找到轮廓
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, area

    cnt = max(contours, key=cv2.contourArea)

    if method == "ellipse" and len(cnt) >= 5:
        # 椭圆拟合（需要 >=5 点）
        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)  # MA: 长轴, ma: 短轴（单位：像素）
        # 注意 OpenCV 返回的是轴长度（直径）
        vertical_d = max(MA, ma) if abs(angle - 90) < 45 else min(MA, ma)
        horizontal_d = min(MA, ma) if abs(angle - 90) < 45 else max(MA, ma)
        # 为了稳健，这里不做角度复杂判断，直接取：
        v_d = max(MA, ma)   # 用长轴近似“垂直”直径（常用于 VCDR）
        h_d = min(MA, ma)   # 用短轴近似“水平”直径（用于 HCDR）
        return float(v_d), float(h_d), area
    else:
        # BBox 方式
        y_idxs, x_idxs = np.where(bin_mask > 0)
        v_d = float(y_idxs.max() - y_idxs.min() + 1)  # 高（近似垂直直径）
        h_d = float(x_idxs.max() - x_idxs.min() + 1)  # 宽（近似水平直径）
        return v_d, h_d, area

# ---------- 主计算函数 ----------
def compute_cdr_from_multiclass_mask(mask: np.ndarray, method: str = "ellipse") -> dict:
    """
    从多类别 mask 计算 CDR
    mask: 0/1/2 或 0/128/255（0=cup, 1=disc, 2=bg）
    返回 dict: {"VCDR":..., "HCDR":..., "area_ratio":..., "cup":{...}, "disc":{...}}
    """
    cls_mask = to_class_index(mask)
    cup  = (cls_mask == 0).astype(np.uint8)
    disc = (cls_mask == 1).astype(np.uint8)

    # 仅保留最大连通域，避免噪点影响
    cup  = largest_component(cup)
    disc = largest_component(disc)

    v_cup, h_cup, a_cup   = diameters_from_mask(cup,  method)
    v_disc, h_disc, a_disc = diameters_from_mask(disc, method)

    # 避免除零
    VCDR = float(v_cup / v_disc) if v_disc > 0 else np.nan
    HCDR = float(h_cup / h_disc) if h_disc > 0 else np.nan
    area_ratio = float(a_cup / a_disc) if a_disc > 0 else np.nan

    return {
        "VCDR": VCDR,
        "HCDR": HCDR,
        "area_ratio": area_ratio,
        "cup":  {"v": v_cup,  "h": h_cup,  "area": a_cup},
        "disc": {"v": v_disc, "h": h_disc, "area": a_disc},
        "method": method
    }


import os
from typing import Optional, Sequence
import cv2
import numpy as np
import pandas as pd
from glob import glob

def to_class_index(mask: np.ndarray) -> np.ndarray:
    """把 0/128/255 调色板映射到 0/1/2；如果已是 0/1/2 直接返回"""
    uniq = set(np.unique(mask).tolist())
    if uniq <= {0, 1, 2}:
        return mask.astype(np.int64)
    out = np.zeros_like(mask, dtype=np.int64)
    out[mask == 0]   = 0
    out[mask == 128] = 1
    out[mask == 255] = 2
    return out

def confusion_binary(gt: np.ndarray, pred: np.ndarray, positive_cls: int):
    gt_pos = (gt == positive_cls)
    pr_pos = (pred == positive_cls)
    TP = np.logical_and(gt_pos,  pr_pos).sum(dtype=np.int64)
    FP = np.logical_and(~gt_pos, pr_pos).sum(dtype=np.int64)
    FN = np.logical_and(gt_pos,  ~pr_pos).sum(dtype=np.int64)
    TN = np.logical_and(~gt_pos, ~pr_pos).sum(dtype=np.int64)
    return TP, FP, FN, TN

def sens_spec_dice(TP, FP, FN, TN, eps=1e-8):
    sens = TP / (TP + FN + eps)
    spec = TN / (TN + FP + eps)
    dice = (2 * TP) / (2 * TP + FP + FN + eps)
    return sens, spec, dice

def calculate_diff_percentage(pred_bin, gt_bin):
    # 计算差异像素数
    diff = np.abs(pred_bin - gt_bin)
    # 百分比
    diff_percentage = diff / gt_bin * 100
    return diff_percentage


def eval_dir(
    gt_dir: str,
    pred_dir: str,
    classes: Sequence[int] = (0, 1, 2),
    save_per_image_csv: Optional[str] = None,
):
    """逐文件名（*.png）对齐，计算每类的敏感度/特异度/Dice"""
    gt_paths = sorted(glob(os.path.join(gt_dir, "*.png")))
    pred_map = {os.path.basename(p): p for p in glob(os.path.join(pred_dir, "*.png"))}

    rows = []
    agg = {c: {"TP":0, "FP":0, "FN":0, "TN":0} for c in classes}
    valid = 0

    for g in gt_paths:
        name = os.path.basename(g)
        if name not in pred_map:
            continue
        gt   = cv2.imread(g, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_map[name], cv2.IMREAD_GRAYSCALE)
        if gt is None or pred is None:
            continue
        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        gt_c   = to_class_index(gt)
        pred_c = to_class_index(pred)

        per_img = {}
        for c in classes:
            TP, FP, FN, TN = confusion_binary(gt_c, pred_c, c)
            s, p, d = sens_spec_dice(TP, FP, FN, TN)
            per_img[f"class{c}_sensitivity"] = s
            per_img[f"class{c}_specificity"] = p
            per_img[f"class{c}_dice"]        = d
            agg[c]["TP"] += TP; agg[c]["FP"] += FP; agg[c]["FN"] += FN; agg[c]["TN"] += TN

        # 每图 macro
        img_macro_sens = float(np.mean([per_img[f"class{c}_sensitivity"] for c in classes]))
        img_macro_spec = float(np.mean([per_img[f"class{c}_specificity"] for c in classes]))
        img_macro_dice = float(np.mean([per_img[f"class{c}_dice"]        for c in classes]))
        row = {"name": name, "macro_sensitivity": img_macro_sens,
               "macro_specificity": img_macro_spec, "macro_dice": img_macro_dice}
        row.update(per_img)
        rows.append(row)
        valid += 1

    if valid == 0:
        raise RuntimeError("没有对齐的文件，请检查 gt_dir / pred_dir 下的 PNG 名称是否一致。")

    per_image_df = pd.DataFrame(rows).sort_values("name")
    if save_per_image_csv:
        per_image_df.to_csv(save_per_image_csv, index=False)
        print(f"[Saved] {save_per_image_csv}")

    # 汇总（micro：像素级累计；macro：对类的平均）
    micro = {}
    for c in classes:
        TP, FP, FN, TN = agg[c]["TP"], agg[c]["FP"], agg[c]["FN"], agg[c]["TN"]
        s, p, d = sens_spec_dice(TP, FP, FN, TN)
        micro[c] = {"sensitivity": float(s), "specificity": float(p), "dice": float(d)}
    macro_sens = float(np.mean([micro[c]["sensitivity"] for c in classes]))
    macro_spec = float(np.mean([micro[c]["specificity"] for c in classes]))
    macro_dice = float(np.mean([micro[c]["dice"]        for c in classes]))

    print(micro)

    print("=== Per-class (micro over pixels) ===")
    for c in classes:
        print(f"Class {c}: Sens={micro[c]['sensitivity']:.4f}  Spec={micro[c]['specificity']:.4f}  Dice={micro[c]['dice']:.4f}")
    print("\n=== Overall (macro over classes) ===")
    print(f"Macro Sensitivity: {macro_sens:.4f}")
    print(f"Macro Specificity: {macro_spec:.4f}")
    print(f"Macro Dice:        {macro_dice:.4f}")




    results_VCDR = []
    results_HCDR = []

    results_cdr_ratio = []

    for g in gt_paths:
        name = os.path.basename(g)
        if name not in pred_map:
            continue
        gt   = cv2.imread(g, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_map[name], cv2.IMREAD_GRAYSCALE)
        if gt is None or pred is None:
            continue
        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        pred_res = compute_cdr_from_multiclass_mask(pred, method="ellipse")  # 或 "bbox"
        gt_res = compute_cdr_from_multiclass_mask(gt, method="ellipse")  # 或 "bbox"
        VCDR = calculate_diff_percentage(pred_res['VCDR'],gt_res['VCDR'])
        HCDR = calculate_diff_percentage(pred_res['HCDR'],gt_res['HCDR'])
        cdr_ratio = calculate_diff_percentage(pred_res['area_ratio'],gt_res['area_ratio'])
        
        
        print("=============")
        print("name",name)
        print("VCDR",VCDR)
        print("HCDR",HCDR)
        
        results_VCDR.append((VCDR+HCDR)/2)

    print("VCDR",np.mean(results_VCDR))

    return per_image_df, micro, {"macro_sensitivity": macro_sens,
                                 "macro_specificity": macro_spec,
                                 "macro_dice": macro_dice,
                                 "macro_cdr": np.mean(results_VCDR)}


def predict(img_path, st):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="train", choices=["train","val_e","infer","eval"])
    ap.add_argument("--weights", type=str, default="/home/yanggq/project/grading/GlaucomaRecognition-main/CodeOfTask3/test3_torch_project/trained_models_torch/best_model_0.9208/model.pt")
    args = ap.parse_args()

    gt_dir = '/home/yanggq/project/grading/task3_disc_cup_segmentation/training/Disc_Cup_Mask'
    out_dir = '/home/yanggq/project/grading/GlaucomaRecognition-main/stream-gamma/pages/funds/result'
    per_image_df, micro, metrics  = eval_to_bmp(args.weights, test_path=img_path, gt_dir=gt_dir, out_dir=out_dir)
    return metrics
