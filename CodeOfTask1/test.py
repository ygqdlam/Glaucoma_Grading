import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 让报错指向真实算子

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models import resnet34, resnet50

import transforms as trans

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline


paddle.set_device('gpu' if paddle.device.is_compiled_with_cuda() else 'cpu')



batchsize = 4 # 4 patients per iter, i.e, 20 steps / epoch
oct_img_size = [512, 512]
image_size = 256
iters = 1000 # For demonstration purposes only, far from reaching convergence
val_ratio = 0.2 # 80 / 20
trainset_root = "/home/yanggq/project/grading/Glaucoma_grading/training/multi-modality_images"
# test_root = ""
num_workers = 4
init_lr = 1e-4
optimizer_type = "adam"


filelists = os.listdir(trainset_root)
train_filelists, val_filelists = train_test_split(filelists, test_size=val_ratio, random_state=12)
print("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))
print(val_filelists)



class GAMMA_sub1_dataset(paddle.io.Dataset):
    """
    getitem() output:
    
    	fundus_img: RGB uint8 image with shape (3, image_size, image_size)
        
        oct_img:    Uint8 image with shape (256, oct_img_size[0], oct_img_size[1])
    """

    def __init__(self,
                 img_transforms,
                 oct_transforms,
                 dataset_root,
                 label_file='',
                 filelists=None,
                 num_classes=3,
                 mode='train'):

        self.dataset_root = dataset_root
        self.img_transforms = img_transforms
        self.oct_transforms = oct_transforms
        self.mode = mode.lower()
        self.num_classes = num_classes
        
        if self.mode == 'train':
            label = {row['data']: row[1:].values 
                        for _, row in pd.read_excel(label_file).iterrows()}

            self.file_list = [[f, label[int(f)]] for f in os.listdir(dataset_root)]
        elif self.mode == "test":
            self.file_list = [[f, None] for f in os.listdir(dataset_root)]
        
        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        real_index, label = self.file_list[idx]
        #print("1111",label)
        fundus_img_path = os.path.join(self.dataset_root, real_index, real_index + ".jpg")
        oct_series_list = sorted(os.listdir(os.path.join(self.dataset_root, real_index, real_index)), 
                                    key=lambda x: int(x.strip("_")[0]))

        fundus_img = cv2.imread(fundus_img_path)[:, :, ::-1] # BGR -> RGB
        oct_series_0 = cv2.imread(os.path.join(self.dataset_root, real_index, real_index, oct_series_list[0]), 
                                    cv2.IMREAD_GRAYSCALE)
        oct_img = np.zeros((len(oct_series_list), oct_series_0.shape[0], oct_series_0.shape[1], 1), dtype="uint8")

        for k, p in enumerate(oct_series_list):
            oct_img[k] = cv2.imread(
                os.path.join(self.dataset_root, real_index, real_index, p), cv2.IMREAD_GRAYSCALE)[..., np.newaxis]

        if self.img_transforms is not None:
            fundus_img = self.img_transforms(fundus_img)
        if self.oct_transforms is not None:
            oct_img = self.oct_transforms(oct_img)
 
        # normlize on GPU to save CPU Memory and IO consuming.
        # fundus_img = (fundus_img / 255.).astype("float32")
        # oct_img = (oct_img / 255.).astype("float32")

        fundus_img = fundus_img.transpose(2, 0, 1) # H, W, C -> C, H, W
        oct_img = oct_img.squeeze(-1) # D, H, W, 1 -> D, H, W

        if self.mode == 'test':
            return fundus_img, oct_img, real_index
        if self.mode == "train":
            #print("222222",label)
            #print("33333333",label.argmax())

            label = label.argmax()
            
            class_id = int(np.argmax(label))      # 先变成 Python int
            class_id = np.int64(class_id)         # 再固化为 np.int64 标量
            print("33333333",class_id)
            
            return fundus_img, oct_img, class_id

    def __len__(self):
        return len(self.file_list)


img_train_transforms = trans.Compose([
    trans.RandomResizedCrop(
        image_size, scale=(0.90, 1.1), ratio=(0.90, 1.1)),
    trans.RandomHorizontalFlip(),
    trans.RandomVerticalFlip(),
    trans.RandomRotation(30)
])

oct_train_transforms = trans.Compose([
    trans.CenterCrop([256] + oct_img_size),
    trans.RandomHorizontalFlip(),
    trans.RandomVerticalFlip()
])

img_val_transforms = trans.Compose([
    trans.CropCenterSquare(),
    trans.Resize((image_size, image_size))
])

oct_val_transforms = trans.Compose([
    trans.CenterCrop([256] + oct_img_size)
])


_train = GAMMA_sub1_dataset(dataset_root=trainset_root, 
                        img_transforms=img_train_transforms,
                        oct_transforms=oct_train_transforms,
                        label_file='/home/yanggq/project/grading/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx')

_val = GAMMA_sub1_dataset(dataset_root=trainset_root, 
                        img_transforms=img_val_transforms,
                        oct_transforms=oct_val_transforms,
                        label_file='/home/yanggq/project/grading/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx')


class Model(nn.Layer):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet34
    """
    def __init__(self):
        super(Model, self).__init__()
        self.fundus_branch = resnet34(pretrained=True, num_classes=0) # remove final fc
        self.oct_branch = resnet34(pretrained=True, num_classes=0) # remove final fc
        self.decision_branch = nn.Linear(512 * 1 * 2, 3) # ResNet34 use basic block, expansion = 1
        
        # replace first conv layer in oct_branch
        self.oct_branch.conv1 = nn.Conv2D(256, 64,
                                        kernel_size=7,
                                        stride=2,
                                        padding=3,
                                        bias_attr=False)

    def forward(self, fundus_img, oct_img):
        b1 = self.fundus_branch(fundus_img)
        b2 = self.oct_branch(oct_img)
        b1 = paddle.flatten(b1, 1)
        b2 = paddle.flatten(b2, 1)
        logit = self.decision_branch(paddle.concat([b1, b2], 1))

        return logit

class Model_resnet50(nn.Layer):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet34
    """
    def __init__(self):
        super(Model_resnet50, self).__init__()
        self.fundus_branch = resnet50(pretrained=True, num_classes=0) # remove final fc
        self.oct_branch = resnet50(pretrained=True, num_classes=0) # remove final fc
        self.decision_branch = nn.Linear(512 * 4 * 2, 3) # ResNet34 use bottleneck block, expansion = 4
        
        # replace first conv layer in oct_branch
        self.oct_branch.conv1 = nn.Conv2D(256, 64,
                                        kernel_size=7,
                                        stride=2,
                                        padding=3,
                                        bias_attr=False)

    def forward(self, fundus_img, oct_img):
        b1 = self.fundus_branch(fundus_img)
        b2 = self.oct_branch(oct_img)
        b1 = paddle.flatten(b1, 1)
        b2 = paddle.flatten(b2, 1)
        logit = self.decision_branch(paddle.concat([b1, b2], 1))

        return logit

img_train_transforms = trans.Compose([
    trans.RandomResizedCrop(
        image_size, scale=(0.90, 1.1), ratio=(0.90, 1.1)),
    trans.RandomHorizontalFlip(),
    trans.RandomVerticalFlip(),
    trans.RandomRotation(30)
])

oct_train_transforms = trans.Compose([
    trans.CenterCrop([256] + oct_img_size),
    trans.RandomHorizontalFlip(),
    trans.RandomVerticalFlip()
])

img_val_transforms = trans.Compose([
    trans.CropCenterSquare(),
    trans.Resize((image_size, image_size))
])

oct_val_transforms = trans.Compose([
    trans.CenterCrop([256] + oct_img_size)
])

train_dataset = GAMMA_sub1_dataset(dataset_root=trainset_root, 
                        img_transforms=img_train_transforms,
                        oct_transforms=oct_train_transforms,
                        filelists=train_filelists,
                        label_file='/home/yanggq/project/grading/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx')

val_dataset = GAMMA_sub1_dataset(dataset_root=trainset_root, 
                        img_transforms=img_val_transforms,
                        oct_transforms=oct_val_transforms,
                        filelists=val_filelists,
                        label_file='/home/yanggq/project/grading/Glaucoma_grading/training/glaucoma_grading_training_GT.xlsx')




train_loader = paddle.io.DataLoader(
    train_dataset,
    batch_sampler=paddle.io.DistributedBatchSampler(train_dataset, batch_size=batchsize, shuffle=True, drop_last=False),
    num_workers=num_workers,
    collate_fn=my_collate,        # ← 新增
    return_list=True,
    use_shared_memory=False
)

val_loader = paddle.io.DataLoader(
    val_dataset,
    batch_sampler=paddle.io.DistributedBatchSampler(val_dataset, batch_size=batchsize, shuffle=True, drop_last=False),
    num_workers=num_workers,
    collate_fn=my_collate,        # ← 新增
    return_list=True,
    use_shared_memory=False
)



model = Model()

if optimizer_type == "adam":
    optimizer = paddle.optimizer.Adam(init_lr, parameters=model.parameters())

criterion = nn.CrossEntropyLoss()



def train(model, iters, train_dataloader, val_dataloader, optimizer, criterion, log_interval, eval_interval):
    iter = 0
    model.train()
    avg_loss_list = []
    avg_kappa_list = []
    best_kappa = 0.
    while iter < iters:
        for data in train_dataloader:
            iter += 1
            if iter > iters:
                break
            fundus_imgs = (data[0] / 255.).astype("float32")
            oct_imgs = (data[1] / 255.).astype("float32")
            labels = data[2].astype('int64')

            logits = model(fundus_imgs, oct_imgs)
            
            # —— 统一 labels：int64 且一维 [N] ——
            if not isinstance(labels, paddle.Tensor):
                labels = paddle.to_tensor(labels)
            if labels.dtype != paddle.int64:
                labels = labels.astype('int64')
            if len(labels.shape) == 2 and labels.shape[-1] == 1:
                labels = paddle.squeeze(labels, axis=-1)
            
            # —— logits/labels 必须在同一设备 ——
            assert logits.place.is_gpu_place() == labels.place.is_gpu_place(), \
                f"device mismatch: logits on {logits.place}, labels on {labels.place}"
            
            # —— 数值检查：logits 不能有 NaN/Inf ——
            if not paddle.isfinite(logits).all():
                bad = int((~paddle.isfinite(logits)).astype('int32').sum().item())
                raise RuntimeError(f"logits contains non-finite values: count={bad}")
            
            # —— 范围检查：labels ∈ [0, C-1]（忽略项除外） ——
            N, C = logits.shape
            assert len(labels.shape) == 1 and labels.shape[0] == N, f"labels shape {list(labels.shape)} vs N={N}"
            lb_min = int(labels.min().item()); lb_max = int(labels.max().item())
            print(f"[debug] N={N}, C={C}, label_min={lb_min}, label_max={lb_max}")
            if lb_min < 0 or lb_max >= C:
                raise ValueError(f"label out of range: min={lb_min}, max={lb_max}, num_classes={C}")
            
            # （若你的数据确实含无效标签，比如 -1/255，则用 ignore_index，并预处理成该值）
            # criterion = paddle.nn.CrossEntropyLoss(ignore_index=-1)
            # labels = paddle.where((labels < 0) | (labels >= C), paddle.to_tensor([-1], dtype='int64')[0], labels)

            loss = criterion(logits, labels)
            # acc = paddle.metric.accuracy(input=logits, label=labels.reshape((-1, 1)), k=1)
            for p, l in zip(logits.numpy().argmax(1), labels.numpy()):
                avg_kappa_list.append([p, l])

            loss.backward()
            optimizer.step()

            model.clear_gradients()
            #avg_loss_list.append(loss.numpy()[0])
            avg_loss_list.append(loss.numpy().item())   # 推荐：numpy 标量 -> Python float

            if iter % log_interval == 0:
                avg_loss = np.array(avg_loss_list).mean()
                avg_kappa_list = np.array(avg_kappa_list)
                avg_kappa = cohen_kappa_score(avg_kappa_list[:, 0], avg_kappa_list[:, 1], weights='quadratic')
                avg_loss_list = []
                avg_kappa_list = []
                print("[TRAIN] iter={}/{} avg_loss={:.4f} avg_kappa={:.4f}".format(iter, iters, avg_loss, avg_kappa))

            if iter % eval_interval == 0:
                avg_loss, avg_kappa = val(model, val_dataloader, criterion)
                print("[EVAL] iter={}/{} avg_loss={:.4f} kappa={:.4f}".format(iter, iters, avg_loss, avg_kappa))
                if avg_kappa >= best_kappa:
                    best_kappa = avg_kappa
                    paddle.save(model.state_dict(),
                            os.path.join('trained_models', "best_model_{:.4f}".format(best_kappa), 'model.pdparams'))
                    paddle.save(optimizer.state_dict(), 
                            os.path.join('trained_models',"best_model_{:.4f}".format(best_kappa), 'optimizer.pdopt'))
                model.train()

def val(model, val_dataloader, criterion):
    model.eval()
    avg_loss_list = []
    cache = []
    with paddle.no_grad():
        for data in val_dataloader:
            fundus_imgs = (data[0] / 255.).astype("float32")
            oct_imgs = (data[1] / 255.).astype("float32")
            labels = data[2].astype('int64')
            
            logits = model(fundus_imgs, oct_imgs)
            for p, l in zip(logits.numpy().argmax(1), labels.numpy()):
                cache.append([p, l])

            loss = criterion(logits, labels)
            # acc = paddle.metric.accuracy(input=logits, label=labels.reshape((-1, 1)), k=1)
            #avg_loss_list.append(loss.numpy()[0])
            avg_loss_list.append(loss.numpy().item())   # 推荐：numpy 标量 -> Python float
    cache = np.array(cache)
    kappa = cohen_kappa_score(cache[:, 0], cache[:, 1], weights='quadratic')
    avg_loss = np.array(avg_loss_list).mean()

    return avg_loss, kappa


fundus, octv, labels = next(train_loader())
print("labels dtype:", labels.dtype, "shape:", labels.shape)
print("labels:", labels, "min/max:", labels.min(), labels.max())
# 期望: dtype=int64, shape=(N,), 值在 [0, C-1]



33333333 0
33333333 0
33333333 333333330
 0
33333333 0
33333333 0
33333333 0
33333333 0
33333333 0
33333333 0
33333333 333333330
 0
33333333 33333333 00

33333333 0
33333333 0
33333333 0
33333333 0
33333333 0
33333333 0
labels dtype: paddle.int64 shape: [4]
labels: Tensor(shape=[4], dtype=int64, place=Place(gpu:0), stop_gradient=True,
       [8672358851485787970, 7375572714207798096, 8391722781239829067,
        8323631284340227184]) min/max: Tensor(shape=[], dtype=int64, place=Place(gpu:0), stop_gradient=True,
       7375572714207798096) Tensor(shape=[], dtype=int64, place=Place(gpu:0), stop_gradient=True,
       8672358851485787970)


        
        
