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
    return pred_idx, pred_png
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


def calculate_diff_percentage(pred_bin, gt_bin):
    # 计算差异像素数
    diff = np.abs(pred_bin - gt_bin)
    # 百分比
    diff_percentage = diff / gt_bin * 100
    return diff_percentage

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


    pred_res = compute_cdr_from_multiclass_mask(pred_idx, method="ellipse")  # 或 "bbox"
    gt_res = compute_cdr_from_multiclass_mask(gt_c, method="ellipse")  # 或 "bbox"
    VCDR = calculate_diff_percentage(pred_res['VCDR'],gt_res['VCDR'])
    HCDR = calculate_diff_percentage(pred_res['HCDR'],gt_res['HCDR'])
    results_VCDR = (VCDR+HCDR)/2


    macro = {
        "macro_sensitivity": float(np.mean([r[1] for r in rows])),
        "macro_specificity": float(np.mean([r[2] for r in rows])),
        "macro_dice":        float(np.mean([r[3] for r in rows])),
        "macro_cdr":         float(results_VCDR),
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



def predict(img_name, st):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",    type=str, default="single", choices=["single"])
    #ap.add_argument("--image",   type=str, required=True, help="输入 RGB 图像路径")
    ap.add_argument("--weights", type=str, default="/home/yanggq/project/grading/GlaucomaRecognition-main/CodeOfTask3/test3_torch_project/trained_models_torch/best_model_0.9208/model.pt", help="模型权重 .pt")
    ap.add_argument("--save",    type=str, default="/home/yanggq/project/grading/GlaucomaRecognition-main/CodeOfTask3/test3_torch_project/result.png", help="输出 PNG（0/128/255）")
    ap.add_argument("--image_dir",      type=str, default="/home/yanggq/project/grading/Glaucoma_grading/training/multi-modality_images", help="可选：GT PNG（0/1/2 或 0/128/255），则计算指标")    
    ap.add_argument("--gt_dir",      type=str, default="/home/yanggq/project/grading/task3_disc_cup_segmentation/training/Disc_Cup_Mask", help="可选：GT PNG（0/1/2 或 0/128/255），则计算指标")
    args, _ = ap.parse_known_args()

    img_path = args.image_dir + '/' + os.path.splitext(img_name)[0] + '/' + img_name
    print(img_name)
    print(args.weights)
    print(args.save)

    pred_idx, mask = infer_single(img_path, args.weights, args.save)

    if args.gt_dir:
        gt_path = find_matching_gt(img_path, args.gt_dir)
        if gt_path is None:
            print(f"[WARN] 在 {args.gt_dir} 中未找到与 {os.path.basename(img_path)} 同名的 GT，跳过评估。")
        # 如果你还留有 --gt 参数，并且找不到同名 GT，可 fallback
        if (gt_path is None) and hasattr(args, "gt") and args.gt:
            gt_path = args.gt if os.path.exists(args.gt) else None

        rows, macro = eval_single(gt_path, pred_idx)
        print("Per-class:")
        for c,s,p,d in rows:
            print(f"  class {c}: Sens={s:.4f}  Spec={p:.4f}  Dice={d:.4f}")
        print("Macro:", macro)

    
    return mask, macro

