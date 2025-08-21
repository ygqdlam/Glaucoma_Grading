# infer_torch.py
import os, re
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score,roc_auc_score
import os, cv2, random, argparse, math, re
import pages.grading.transforms as trans
from torchvision.models import resnet34



# ==== 配置 ====
image_size   = 256
oct_img_size = [512, 512]
num_classes  = 3


# ==== 轻量 transforms（与你之前的一致风格）====
img_test_transforms = trans.Compose([
    trans.CropCenterSquare(),
    trans.Resize((image_size, image_size)),
])
oct_test_transforms = trans.Compose([
    trans.CenterCrop([256] + oct_img_size),
])

# ==== 测试数据集 ====
class GAMMA_sub1_dataset_test(Dataset):
    """
    输出:
      fundus_img: uint8, (3, H, W)  RGB
      oct_img   : uint8, (D, H, W)  灰度体
      idx       : 样本 ID (str)
    """
    def __init__(self, dataset_root, img_transforms=None, oct_transforms=None,filelists=None):
        self.dataset_root   = dataset_root
        self.img_transforms = img_transforms
        self.oct_transforms = oct_transforms
        self.file_list = filelists

    def _oct_sort_key(self, name: str):
        stem = os.path.splitext(name)[0]
        m = re.search(r'(\d+)$', stem)
        return int(m.group(1)) if m else stem

    def __getitem__(self, idx):
        real_index = self.file_list[idx]
        fundus_img_path = os.path.join(self.dataset_root, real_index, real_index + ".jpg")
        series_dir = os.path.join(self.dataset_root, real_index, real_index)
        oct_series_list = sorted(os.listdir(series_dir), key=self._oct_sort_key)

        # 读图
        fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1]  # BGR->RGB
        oct0 = cv2.imread(os.path.join(series_dir, oct_series_list[0]), cv2.IMREAD_GRAYSCALE)
        D, H, W = len(oct_series_list), oct0.shape[0], oct0.shape[1]
        oct_img = np.zeros((D, H, W, 1), dtype="uint8")
        for k, p in enumerate(oct_series_list):
            oct_img[k] = cv2.imread(os.path.join(series_dir, p), cv2.IMREAD_GRAYSCALE)[..., np.newaxis]

        # transforms
        if self.img_transforms is not None:
            fundus_img = self.img_transforms(fundus_img)
        if self.oct_transforms is not None:
            oct_img = self.oct_transforms(oct_img)

        # NHWC->CHW / DHWC->DHW
        fundus_img = fundus_img.transpose(2, 0, 1)
        oct_img    = oct_img.squeeze(-1)

        return fundus_img, oct_img, real_index

    def __len__(self):
        return len(self.file_list)

# ===== collate（显式堆叠，保持 dtype）=====
def collate_test(batch):
    f_list, o_list, idx_list = zip(*batch)
    f = np.stack(f_list, axis=0).astype("uint8")  # [N,3,H,W]
    o = np.stack(o_list, axis=0).astype("uint8")  # [N,D,H,W]
    return f, o, list(idx_list)

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

# ===== 推理 =====
def tensorize_batch(batch, device):
    f, o, idx_list = batch
    fundus = torch.from_numpy(f).float().div_(255.0).to(device, non_blocking=True)  # [N,3,H,W]
    octv   = torch.from_numpy(o).float().div_(255.0).to(device, non_blocking=True)  # [N,D,H,W]
    return fundus, octv, idx_list

def main(testset_root, gt_df, best_model_path):
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ========= 划分数据 =========
    filelists = sorted(os.listdir(testset_root))
    print("filelists",filelists)
    #train_filelists, val_filelists = train_test_split(filelists, test_size=val_ratio, random_state=12)

    # 最后 20 个作为测试集
    val_filelists = filelists[-20:]
    # 其余的作为训练集
    train_filelists = filelists[:-20]

    print(f"Total Nums: {len(filelists)}, train: {len(train_filelists)}, val: {len(val_filelists)}")
    print("val_filelists",val_filelists)
    # 数据
    test_dataset = GAMMA_sub1_dataset_test(
        dataset_root=testset_root,
        img_transforms=img_test_transforms,
        oct_transforms=oct_test_transforms,
        filelists=val_filelists,
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,
                             num_workers=2, collate_fn=collate_test, pin_memory=True)

    # 模型 & 权重
    model = Model(num_classes=num_classes).to(device)
    state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in test_loader:
            fundus, octv, idx_list = tensorize_batch(batch, device)
            logits = model(fundus, octv)                   # [N,3]
            preds  = torch.argmax(logits, dim=1).cpu().numpy()  # [N]

            # 组装提交格式
            for i, idx in enumerate(idx_list):
                p = int(preds[i])
                rows.append([
                    idx,
                    int(p == 0),  # non
                    int(p == 1),  # early
                    int(p == 2),  # mid_advanced
                ])
    pred_df = pd.DataFrame(rows, columns=["data", "non", "early", "mid_advanced"])


    # 提取标签（取最大概率对应的类别）
    print(gt_df)
    print(pred_df)

    gt_labels = gt_df[['non', 'early', 'mid_advanced']].values.argmax(axis=1)
    pred_labels = pred_df[['non', 'early', 'mid_advanced']].values.argmax(axis=1)
    # 计算 Cohen's Kappa
    kappa = cohen_kappa_score(gt_labels, pred_labels)
    print("Cohen's Kappa:", kappa)



    # 提取真实标签和预测概率
    y_true = gt_df[['non', 'early', 'mid_advanced']].values
    y_pred = pred_df[['non', 'early', 'mid_advanced']].values
    # 计算 macro-AUC
    macro_auc = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")
    print("Macro-AUC:", macro_auc)

    from sklearn.metrics import accuracy_score
    # 计算准确率
    accuracy = accuracy_score(gt_labels, pred_labels)
    print("Accuracy:", accuracy * 100)


    return accuracy * 100, kappa, macro_auc



def predict(img_name, st):
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",    type=str, default="single", choices=["single"])
    ap.add_argument("--dataset_root",    type=str, default="/home/yanggq/project/grading/Glaucoma_grading/training/multi-modality_images", choices=["single"])
    ap.add_argument("--image_dir",      type=str, default="/home/yanggq/project/grading/Glaucoma_grading/training/multi-modality_images", help="可选：GT PNG（0/1/2 或 0/128/255），则计算指标")    

    ap.add_argument("--weights", type=str, default="/home/yanggq/project/grading/GlaucomaRecognition-main/CodeOfTask1/trained_models_torch/best_model_0.7523/model.pt", help="模型权重 .pt")
    ap.add_argument("--gt_excel",      type=str, default="/home/yanggq/project/grading/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx", help="可选：GT PNG（0/1/2 或 0/128/255），则计算指标")
    args, _ = ap.parse_known_args()  # 兼容 Jupyter
    gt_df = pd.read_excel(args.gt_excel)  # predictions

    acc, kappa, macro_auc = main(img_name, gt_df, args.weights)

    return acc, kappa, macro_auc
