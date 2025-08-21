# test1_torch.py
import os, cv2, argparse,re
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import cohen_kappa_score,roc_auc_score
from torchvision.models import resnet34

def _oct_sort_key(name: str):
    stem = os.path.splitext(name)[0]
    m = re.search(r'(\d+)$', stem)
    return int(m.group(1)) if m else stem

def load_single_sample(dataset_root: str,
                       sample_id: str,
                       img_transforms=None,
                       oct_transforms=None):
    """
    读取一个样本并返回：
      fundus_img: uint8, (3, H, W)  RGB (CHW)
      oct_img   : uint8, (D, H, W)  灰度体 (DHW)
      idx       : 样本 ID (str)
    路径结构与原 Dataset 完全一致：
      dataset_root/sample_id/sample_id.jpg
      dataset_root/sample_id/sample_id/*.png (OCT 序列)
    """

    # fundus 路径
    fundus_img_path = os.path.join(dataset_root, sample_id, f"{sample_id}.jpg")
    if not os.path.exists(fundus_img_path):
        raise FileNotFoundError(f"fundus not found: {fundus_img_path}")

    # OCT 目录
    series_dir = os.path.join(dataset_root, sample_id, sample_id)
    if not os.path.isdir(series_dir):
        raise FileNotFoundError(f"OCT folder not found: {series_dir}")

    # OCT 文件排序（与原逻辑一致）
    oct_series_list = sorted(os.listdir(series_dir), key=_oct_sort_key)
    oct_series_list = [p for p in oct_series_list
                       if os.path.isfile(os.path.join(series_dir, p))]
    if len(oct_series_list) == 0:
        raise RuntimeError(f"No OCT slices found in: {series_dir}")

    # 读 fundus（BGR->RGB）
    fundus_img = cv2.imread(fundus_img_path, cv2.IMREAD_COLOR)
    if fundus_img is None:
        raise RuntimeError(f"cv2 failed to read: {fundus_img_path}")
    fundus_img = fundus_img[:, :, ::-1]  # BGR -> RGB

    # 读第一张 OCT，获取尺寸
    oct0 = cv2.imread(os.path.join(series_dir, oct_series_list[0]), cv2.IMREAD_GRAYSCALE)
    if oct0 is None:
        raise RuntimeError(f"cv2 failed to read first OCT slice in: {series_dir}")
    H, W = oct0.shape[:2]
    D = len(oct_series_list)

    # 构建体数据 (D,H,W,1) -> 稍后 squeeze 到 (D,H,W)
    oct_img = np.zeros((D, H, W, 1), dtype="uint8")
    for k, p in enumerate(oct_series_list):
        sl = cv2.imread(os.path.join(series_dir, p), cv2.IMREAD_GRAYSCALE)
        if sl is None:
            raise RuntimeError(f"cv2 failed to read OCT slice: {p}")
        oct_img[k] = sl[..., np.newaxis]

    # 可选：transforms（与 Dataset 的行为一致）
    if img_transforms is not None:
        fundus_img = img_transforms(fundus_img)  # 期望 NHWC 输入
    if oct_transforms is not None:
        oct_img = oct_transforms(oct_img)        # 期望 DHWC 输入

    # NHWC->CHW / DHWC->DHW
    if fundus_img.ndim == 3 and fundus_img.shape[2] == 3:
        fundus_img = fundus_img.transpose(2, 0, 1)  # (H,W,3) -> (3,H,W)
    else:
        raise ValueError(f"Unexpected fundus_img shape after transform: {fundus_img.shape}")

    if oct_img.ndim == 4 and oct_img.shape[-1] == 1:
        oct_img = oct_img.squeeze(-1)  # (D,H,W,1) -> (D,H,W)
    elif oct_img.ndim == 3:
        # 已经是 (D,H,W)
        pass
    else:
        raise ValueError(f"Unexpected oct_img shape after transform: {oct_img.shape}")

    return fundus_img, oct_img, sample_id


# ========= 你的模型（如原文件已有模型，保留或替换为同名类） =========
class Model(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.fundus_branch = resnet34(weights="IMAGENET1K_V1")
        self.oct_branch    = resnet34(weights="IMAGENET1K_V1")

        # 去掉 fc，只取 512 维特征
        self.fundus_branch.fc = nn.Identity()
        self.oct_branch.fc    = nn.Identity()

        # OCT 分支第一层改为 256 输入通道（把 D 当通道）
        self.oct_branch.conv1 = nn.Conv2d(256, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.fc = nn.Linear(512 * 2, num_classes)

    def forward(self, fundus_img, oct_img):
        b1 = self.fundus_branch(fundus_img)  # [N,512]
        b2 = self.oct_branch(oct_img)        # [N,512]
        return self.fc(torch.cat([b1, b2], dim=1))  # [N, num_classes]
    
# ========= 工具函数 =========
IMAGE_SIZE = 512  # 和训练一致

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
    sens=TP/(TP+FN+eps)
    spec=TN/(TN+FP+eps)
    dice=(2*TP)/(2*TP+FP+FN+eps)
    return sens,spec,dice

# ========= 单张推理 =========
@torch.no_grad()
def infer_single(dataset_root: str, image_path: str, weights_path: str, device: Optional[str]=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    name, _ = os.path.splitext(os.path.basename(image_path))


    fundus, octvol, idx = load_single_sample(dataset_root, name,
                                             None, None)
    # 转 float 并归一化到 [0,1]（与原训练一致）
    fundus = (fundus / 255.).astype("float32")  # (3,H,W)
    octvol = (octvol / 255.).astype("float32")  # (D,H,W)

    # 加 batch 维
    fundus_t = torch.from_numpy(fundus).unsqueeze(0).to(device)  # [1,3,H,W]
    octvol_t = torch.from_numpy(octvol).unsqueeze(0).to(device)  # [1,D,H,W] 或按你的网络期望维度
    model = Model(num_classes=3).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)

    model.eval()
    logits = model(fundus_t, octvol_t)   # 形状例如 [1, num_classes]
    # 如果是分类任务：
    #probs = torch.softmax(logits, dim=1)
    #pred  = probs.argmax(1).item()       # int
    #return idx, pred, probs.squeeze(0).cpu().numpy()

    pred_idx = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
    pred = int(pred_idx)
    # 构造单行结果
    rows = [[
        idx,
        int(pred == 0),  # non
        int(pred == 1),  # early
        int(pred == 2),  # mid_advanced
    ]]

    # 生成 DataFrame
    pred_df = pd.DataFrame(rows, columns=["data", "non", "early", "mid_advanced"])

    return pred_df

def eval_single(gt_path: str, pred_idx: np.ndarray, classes=(0,1,2)):
    #gt = gt_path[["non", "early", "mid_advanced"]].values.astype(int)
    #pred = pred_idx[["non", "early", "mid_advanced"]].values.astype(int)

    gt = gt_path[["non", "early", "mid_advanced"]].astype(int)
    pred = pred_idx[["non", "early", "mid_advanced"]].astype(int)
    print(pred)
    print(gt)
    return gt, pred



def find_row_by_image(
    image_path: str,
    excel_path: str,
    possible_cols = ("image", "image_name", "filename", "file", "name", "data", "ID")
):
    """
    在 Excel 中按图像名查找一行：
    - 自动读取 Excel
    - 自动选择最可能的“文件名/ID”列
    - 支持 image_path 带或不带扩展名；Excel 中也可带/不带扩展名
    - 大小写不敏感，自动去空白

    返回：pd.Series（匹配到的第一行）；如有多个匹配，返回第一条并打印提示；找不到返回 None
    """
    # 规范化目标名（stem 优先）
    target_full = os.path.basename(image_path)           # e.g. "0001.jpg"
    target_stem = os.path.splitext(target_full)[0]       # e.g. "0001"

    # 读取 Excel
    df = pd.read_excel(excel_path, dtype=str)  # 自动引擎；如需指定：engine="openpyxl"
    # 统一列名
    df.columns = [str(c).strip() for c in df.columns]
    # 寻找最可能作为“文件名/ID”的列
    key_col = None
    for c in possible_cols:
        if c in df.columns:
            key_col = c
            break
        # 宽松匹配（不严格要求完全一致）
        for col in df.columns:
            if c.lower() == col.lower():
                key_col = col
                break
        if key_col: break

    if key_col is None:
        raise ValueError(
            f"Excel 中未找到可作为文件名/ID 的列。已尝试：{possible_cols}；实际列：{list(df.columns)}"
        )

    # 构造一个标准化的比较列（去空格、转小写）
    col_norm = df[key_col].astype(str).str.strip().str.lower()
    # 两种形式：带扩展名 vs 不带扩展名
    match_mask = (col_norm == target_full.lower()) | (col_norm == target_stem.lower())
    matches = df[match_mask]

    if len(matches) == 0:
        # 再宽松一点：如果 Excel 里存的是 “0001.png”，而 image 是 “0001.JPG” 等
        # 尝试去拓展名再比
        col_noext = col_norm.str.replace(r"\.[^.]+$", "", regex=True)
        match_mask2 = (col_noext == target_stem.lower())
        matches = df[match_mask2]

    if len(matches) == 0:
        print(f"[WARN] Excel 未找到与图像名匹配的行：{target_full} / {target_stem}")
        return None

    if len(matches) > 1:
        print(f"[INFO] 匹配到多行，返回第一行。匹配数量：{len(matches)}")
    #return matches.iloc[0]  # 返回第一条匹配行（pd.Series）
    return matches  # 返回第一条匹配行（pd.Series）

# ========= CLI =========
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",    type=str, default="single", choices=["single"])
    ap.add_argument("--dataset_root",    type=str, default="/home/yanggq/project/grading/Glaucoma_grading/training/multi-modality_images", choices=["single"])

    ap.add_argument("--image",   type=str, required=True, help="输入 RGB 图像路径")
    ap.add_argument("--weights", type=str, default="/home/yanggq/project/grading/GlaucomaRecognition-main/CodeOfTask1/trained_models_torch/best_model_0.7523/model.pt", help="模型权重 .pt")
    ap.add_argument("--gt_excel",      type=str, default="/home/yanggq/project/grading/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx", help="可选：GT PNG（0/1/2 或 0/128/255），则计算指标")
    args, _ = ap.parse_known_args()  # 兼容 Jupyter
    return args

### python test1_single_torch.py --image /home/yanggq/project/grading/Glaucoma_grading/training/multi-modality_images/0001/0001.jpg

if __name__ == "__main__":
    import numpy as np
    args = get_args()
    pred_idx = infer_single(args.dataset_root, args.image, args.weights)
    if args.gt_excel:
        gt_idx = find_row_by_image(args.image, args.gt_excel)
        gt, pred = eval_single(gt_idx, pred_idx)
        
