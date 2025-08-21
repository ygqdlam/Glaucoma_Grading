# test3_torch.py
import os, cv2, argparse
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


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

# ========= 工具 =========
IMAGE_SIZE = 512

def center_crop(img: np.ndarray, size_hw: Tuple[int,int]) -> np.ndarray:
    h,w = img.shape[:2]
    th,tw = size_hw
    th=min(th,h); tw=min(tw,w)
    i=(h-th)//2; j=(w-tw)//2
    return img[i:i+th, j:j+tw]

def resize_image(img: np.ndarray, size_hw: Tuple[int,int], is_mask: bool=False) -> np.ndarray:
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(img, (size_hw[1], size_hw[0]), interpolation=interp)

def to_class_index(mask: np.ndarray) -> np.ndarray:
    uniq = set(np.unique(mask).tolist())
    if uniq <= {0,1,2}:
        return mask.astype(np.int64)
    out = np.zeros_like(mask, dtype=np.int64)
    out[mask==0]=0; out[mask==128]=1; out[mask==255]=2
    return out

def to_palette(mask_idx: np.ndarray) -> np.ndarray:
    out = mask_idx.astype(np.uint8).copy()
    out[mask_idx==1]=128
    out[mask_idx==2]=255
    return out

def confusion_binary(gt: np.ndarray, pred: np.ndarray, positive_cls: int):
    gt_pos=(gt==positive_cls); pr_pos=(pred==positive_cls)
    TP=np.logical_and(gt_pos, pr_pos).sum(dtype=np.int64)
    FP=np.logical_and(~gt_pos, pr_pos).sum(dtype=np.int64)
    FN=np.logical_and(gt_pos, ~pr_pos).sum(dtype=np.int64)
    TN=np.logical_and(~gt_pos, ~pr_pos).sum(dtype=np.int64)
    return TP,FP,FN,TN

def sens_spec_dice(TP,FP,FN,TN,eps=1e-8):
    sens=TP/(TP+FN+eps); spec=TN/(TN+FP+eps); dice=(2*TP)/(2*TP+FP+FN+eps)
    return sens,spec,dice

# ========= 单张推理 =========
@torch.no_grad()
def infer_single(image_path: str, weights_path: str, save_png_path: str, device: Optional[str]=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h0,w0 = img_rgb.shape[:2]
    s = min(h0,w0)
    img_sq  = center_crop(img_rgb, (s,s))
    img_net = resize_image(img_sq, (IMAGE_SIZE, IMAGE_SIZE), is_mask=False)

    x = img_net.astype(np.float32).transpose(2,0,1)/255.0
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    model = CupDiscUNet(num_classes=3).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    logits = model(x)
    pred_idx = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    pred_png = to_palette(pred_idx)
    os.makedirs(os.path.dirname(save_png_path) or ".", exist_ok=True)
    cv2.imwrite(save_png_path, pred_png, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"[Saved] {save_png_path}")
    return pred_idx

def eval_single(gt_path: str, pred_idx: np.ndarray, classes=(0,1,2)):
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(gt_path)
    gt_c = to_class_index(gt)
    if gt_c.shape != pred_idx.shape:
        pred_idx = cv2.resize(pred_idx, (gt_c.shape[1], gt_c.shape[0]), interpolation=cv2.INTER_NEAREST)

    rows=[]
    for c in classes:
        TP,FP,FN,TN = confusion_binary(gt_c, pred_idx, positive_cls=c)
        s,p,d = sens_spec_dice(TP,FP,FN,TN)
        rows.append((c,s,p,d))
    macro = {
        "macro_sensitivity": float(np.mean([r[1] for r in rows])),
        "macro_specificity": float(np.mean([r[2] for r in rows])),
        "macro_dice":        float(np.mean([r[3] for r in rows])),
    }
    return rows, macro

def find_matching_gt(image_path: str, gt_dir: str,
                     exts=(".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")) -> str | None:
    """
    在 gt_dir 中按“与 image 同名（不含扩展名）”查找 GT 文件。
    优先匹配 .png；找不到则按 exts 依次尝试。
    返回：GT 的完整路径；找不到返回 None。
    """
    stem = Path(image_path).stem
    gt_png = os.path.join(gt_dir, stem + ".png")
    if os.path.exists(gt_png):
        return gt_png
    for ext in exts:
        p = os.path.join(gt_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None

# ========= CLI =========
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",    type=str, default="single", choices=["single"])
    ap.add_argument("--image",   type=str, required=True, help="输入 RGB 图像路径")
    ap.add_argument("--weights", type=str, default="/home/yanggq/project/grading/GlaucomaRecognition-main/CodeOfTask3/test3_torch_project/trained_models_torch/best_model_0.9208/model.pt", help="模型权重 .pt")
    ap.add_argument("--save",    type=str, default="/home/yanggq/project/grading/GlaucomaRecognition-main/CodeOfTask3/test3_torch_project/result.png", help="输出 PNG（0/128/255）")
    ap.add_argument("--gt_dir",      type=str, default="/home/yanggq/project/grading/task3_disc_cup_segmentation/training/Disc_Cup_Mask", help="可选：GT PNG（0/1/2 或 0/128/255），则计算指标")
    args, _ = ap.parse_known_args()
    return args



if __name__ == "__main__":
    args = get_args()
    pred_idx = infer_single(args.image, args.weights, args.save)
    if args.gt_dir:
        gt_path = find_matching_gt(args.image, args.gt_dir)
        if gt_path is None:
            print(f"[WARN] 在 {args.gt_dir} 中未找到与 {os.path.basename(args.image)} 同名的 GT，跳过评估。")
        # 如果你还留有 --gt 参数，并且找不到同名 GT，可 fallback
        if (gt_path is None) and hasattr(args, "gt") and args.gt:
            gt_path = args.gt if os.path.exists(args.gt) else None

        rows, macro = eval_single(gt_path, pred_idx)
        print("Per-class:")
        for c,s,p,d in rows:
            print(f"  class {c}: Sens={s:.4f}  Spec={p:.4f}  Dice={d:.4f}")
        print("Macro:", macro)
