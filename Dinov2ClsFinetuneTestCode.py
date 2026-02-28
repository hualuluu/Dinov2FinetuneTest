"""
用于做dinov2 分类的实验代码
test 1: 参考代码 - 微调 backbone + cls , 加载采用 torch.hub.load 
test 2: 参考代码 - 微调 backbone + cls , 加载采用 本地的model param 
test 3: 参考代码 - 仅训练分类头 cls 冻结backbone, 加载采用 torch.hub.load 
test 4: 参考代码 - 仅训练分类头 cls 冻结backbone, 加载采用 本地的model param 

为了后续 转C++ 先采用 Opencv 加载图像
包括：数据加载， 分类头， 测试代码

getDinov2Vits14Model 中vit_model的构造方式修改了一下，不采用vits.__dict__定义
因为vits.__dict__中指定了MemEffAttention结构，这个结构涉及到 xformers版本问题 导致 转onnx 有问题
我自己的代码是由cuda要求的，所以采用基础的DinoVisionTransformer加载网络结构，指定的是Attention结构
"""

# 1. 导入并忽略所有警告（可选）
import torch, warnings, os, cv2, random, math
warnings.filterwarnings("ignore")
# cuda 设置
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

from sklearn.metrics import classification_report, confusion_matrix
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,  transforms
from dinov2.models.vision_transformer import DinoVisionTransformer
import torch.optim as optim
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# 为 CPU 设置随机种子
torch.manual_seed(42)
# 为当前 GPU 设置种子
torch.cuda.manual_seed(42)
# 如果使用多块 GPU，为所有 GPU 设置种子
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)

PIXEL_MEANS = (0.485, 0.456, 0.406)    # RGB  format mean and variances
PIXEL_STDS = (0.229, 0.224, 0.225)

# 构建数据集
class GetClsDatasets(Dataset):
    def __init__(self, dataDir, train = True, inputw=224, inputh=224):
        self.imageroot = dataDir
        self.train = train
        self.inputw = inputw
        self.inputh = inputh
        self.imageData = []
        fileList = os.listdir(self.imageroot)
        for filename in fileList:
            fileroot = os.path.join(self.imageroot, filename)
            imagelist = os.listdir(fileroot)
            cls_id = filename.split('_')[-1]
            for imagename in imagelist:
                imagepath = os.path.join(fileroot, imagename)
                self.imageData.append([imagepath, cls_id])

    def __len__(self):
        return len(self.imageData)

    def __getitem__(self, id):
        imagepath = self.imageData[id][0]
        image = cv2.imread(imagepath)
        cls_id = int(self.imageData[id][1])
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 先试试简单的 resize 
        image = cv2.resize(imageRGB, (self.inputw, self.inputh), interpolation=cv2.INTER_LINEAR)
        
        # 4. 转换为 float32 并缩放到 [0, 1]（等效 ToTensor）
        img_float = image.astype(np.float32) / 255.0

        # 5. 归一化 (Normalize)
        # 将 mean 和 std 转换为 numpy 数组并 reshape 以适配广播
        mean_arr = np.array(PIXEL_MEANS, dtype=np.float32).reshape(1, 1, 3)
        std_arr = np.array(PIXEL_STDS, dtype=np.float32).reshape(1, 1, 3)
        img_norm = (img_float - mean_arr) / std_arr

        # 6. 将 HWC 转换为 CHW（PyTorch 张量格式）
        img_chw = np.transpose(img_norm, (2, 0, 1))
        imageTensor = torch.from_numpy(img_chw)

        return imageTensor, torch.tensor(cls_id)
        

def getDinov2ClsDatasets(dataDir, dataFlag = "transform", imageSize = 518,
                          batchSize = 64, mean = PIXEL_MEANS, std = PIXEL_STDS):
    if dataFlag == "transform":
        # 准备数据
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((imageSize,imageSize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize((imageSize,imageSize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
        }
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(dataDir, x), data_transforms[x]) 
            for x in ['train', 'val']
        }
        
        data_loaders = {
            x: DataLoader(image_datasets[x], shuffle=True, batch_size=batchSize, num_workers=4)
            for x in ['train', 'val']
        }

        return data_loaders["train"] , data_loaders["val"]

    else:
        valroot = os.path.join(dataDir, "val")
        ValClsDataset = GetClsDatasets(valroot, True, imageSize, imageSize)
        valLoader= DataLoader(ValClsDataset, batch_size=batchSize, shuffle=True, num_workers=4)
        print(f"Validation samples: {len(ValClsDataset)}")

        trainroot = os.path.join(dataDir, "train")
        TrainClsDataset = GetClsDatasets(trainroot, True, imageSize, imageSize)
        trainLoader = DataLoader(TrainClsDataset, batch_size=batchSize, shuffle=True,  num_workers=4)
        print(f"Training samples: {len(TrainClsDataset)}")
    
        return trainLoader , valLoader


# 构建模型
class Dinov2ClsModel(nn.Module):
    def __init__(self, dinov2Vits14Model, classNum):
        super().__init__()
        self.transformer = dinov2Vits14Model
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, classNum)
        )

    def forward(self, x):
        features = self.transformer.forward_features(x)
        # 提取 [CLS] token 特征（归一化后的）
        cls_token = features['x_norm_clstoken']  # 形状: [B, hidden_dim]
        x = self.classifier(cls_token)
        return x
    

# 比较模型参数
def compare_state_dicts(state_dict1, state_dict2, threshold=1e-6):
    """
    比较两个 state_dict，输出参数发生变化的层及变化幅度。
    
    Args:
        state_dict1 (dict): 第一个 epoch 的权重字典
        state_dict2 (dict): 第二个 epoch 的权重字典
        threshold (float): 判断参数是否“改变”的阈值（基于最大绝对值差异）
    
    Returns:
        dict: 包含变化层信息的字典，键为层名，值为 (L2距离, 最大差异)
    """
    changed_layers = {}
    all_keys = set(state_dict1.keys()) | set(state_dict2.keys())
    
    for key in all_keys:
        if key not in state_dict1:
            print(f"Warning: {key} not found in first state_dict")
            continue
        if key not in state_dict2:
            print(f"Warning: {key} not found in second state_dict")
            continue
        
        param1 = state_dict1[key]
        param2 = state_dict2[key]
        
        # 确保形状相同
        if param1.shape != param2.shape:
            print(f"Shape mismatch for {key}: {param1.shape} vs {param2.shape}")
            continue
        
        # 计算差异
        diff = param1 - param2
        l2_dist = torch.norm(diff).item()
        max_diff = torch.max(torch.abs(diff)).item()
        
        if max_diff > threshold:
            changed_layers[key] = (l2_dist, max_diff)
    
    return changed_layers


# 为了保证loacal模式 和hub模式下一样可以支持任意输入尺寸图像
def adapt_position_encoding(model, new_img_size=518, patch_size=14):
    """
    调整 DINOv2 模型的位置编码以适配新的输入尺寸。
    
    Args:
        model: DINOv2 ViT 模型（应包含 pos_embed 属性）
        new_img_size: 新的正方形图像边长（像素）
        patch_size: 模型的 patch 大小（DINOv2 默认为 14）
    """
    # 获取旧的位置编码
    old_pos_embed = model.pos_embed  # 形状: [1, old_seq_len, dim]
    old_seq_len = old_pos_embed.shape[1]
    embed_dim = old_pos_embed.shape[-1]
    
    # 计算旧的 patch 网格大小
    old_num_patches = (model.patch_embed.img_size[0] // patch_size) ** 2
    # 额外的 token 数量（如 [CLS]、寄存器等）
    num_extra_tokens = old_seq_len - old_num_patches
    
    # 计算新的 patch 网格大小和序列长度
    new_grid_size = new_img_size // patch_size
    new_num_patches = new_grid_size ** 2
    new_seq_len = new_num_patches + num_extra_tokens
    
    if old_seq_len == new_seq_len:
        print("位置编码尺寸已匹配，无需调整。")
        return
    
    print(f"调整位置编码: 从 {old_seq_len} 到 {new_seq_len}")
    
    # 分离固定 token 的位置编码（前 num_extra_tokens 个）
    extra_pos_embed = old_pos_embed[:, :num_extra_tokens, :]  # [1, num_extra_tokens, dim]
    
    # 分离图像块的位置编码
    patch_pos_embed = old_pos_embed[:, num_extra_tokens:, :]  # [1, old_num_patches, dim]
    
    # 将图像块位置编码从 1D 序列重塑为 2D 空间网格
    old_grid_size = int(math.sqrt(old_num_patches))
    # 形状: [1, dim, old_grid_size, old_grid_size]
    patch_pos_embed_2d = patch_pos_embed.reshape(1, old_grid_size, old_grid_size, embed_dim).permute(0, 3, 1, 2)
    
    # 使用双三次插值缩放到新的网格尺寸
    new_patch_pos_embed_2d = F.interpolate(
        patch_pos_embed_2d,
        size=(new_grid_size, new_grid_size),
        mode='bicubic',
        align_corners=False
    )
    
    # 将插值后的 2D 网格重新展平为 1D 序列
    # 形状: [1, new_num_patches, dim]
    new_patch_pos_embed = new_patch_pos_embed_2d.permute(0, 2, 3, 1).reshape(1, new_num_patches, embed_dim)
    
    # 拼接固定 token 和新的图像块位置编码
    new_pos_embed = torch.cat([extra_pos_embed, new_patch_pos_embed], dim=1)
    
    # 替换模型中的位置编码
    model.pos_embed = torch.nn.Parameter(new_pos_embed)
    print("位置编码调整完成。")


# modelFlag = "local" / "hub"
def getDinov2Vits14Model(modelFlag = "local", imageSize = 224, patchSize = 14,model_size = "vits14"):

    vit_model= None
    if(modelFlag == "local"):
        
        # vit_model = vits.vit_small(img_size = imageSize, patch_size=patchSize)
        """
        vit_model = vits.__dict__['vit_small'](
            img_size=518,#imageSize,
            patch_size=patchSize,
            num_register_tokens=0,          # 官方模型有4个寄存器
            init_values=1e-5,                # LayerScale初始值
            block_chunks=0,                   # 0表示不分组
        )
        """
        model_map = {
            "vits14": ("vit_small", 384),
            "vitb14": ("vit_base", 768), 
            "vitl14": ("vit_large", 1024),
            "vitg14": ("vit_giant2", 1536)
        }
    
        arch_name, embed_dim = model_map[model_size]

    
        vit_model = DinoVisionTransformer(
            img_size = 518,
            patch_size=14,
            embed_dim=embed_dim,
            depth=12 if "small" in arch_name else 12 if "base" in arch_name else 24,
            num_heads=6 if "small" in arch_name else 12 if "base" in arch_name else 16,
            mlp_ratio=4,
            num_register_tokens=0,
            init_values=1e-5,
            block_chunks=0                   # 0表示不分组
        )
        state_dict = torch.load("dinov2_vits14_pretrain.pth", map_location='cpu')
        # 直接加载，使用 strict=False 查看缺失/多余键
        msg = vit_model.load_state_dict(state_dict, strict=False)
        print("Missing keys:", msg.missing_keys)
        print("Unexpected keys:", msg.unexpected_keys)

        # 调整位置编码到目标尺寸 imageSize
        adapt_position_encoding(vit_model, new_img_size=imageSize, patch_size=patchSize)

    elif(modelFlag == "hub"):
        vit_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    if vit_model != None:
        print("vit_small Backbone loaded successfully!")

    return vit_model



# 测试流程
def test(valLoader, modelpath, class_names = []):
    
    
    modelDinov2Cls = torch.load(modelpath).to(device)
    modelDinov2Cls.eval()
    total = 0.0
    correct = 0.0
    val_predicted = []
    val_labels = []
    with torch.no_grad():
        for image, clss in valLoader:
            #print(image.shape, clss)
            image, target = image.to(device), clss.to(device)

            outputs = modelDinov2Cls(image).to(device)
    
            _, predicted = torch.max(outputs.data, 1)

            total += target.size(0)
            correct += (predicted.to(device) == target).sum().item()
            
            val_labels += (target.cpu().numpy().tolist())
            val_predicted += (predicted.cpu().numpy().tolist())
    
    print(f'Accuracy of the network on the {len(valLoader)*6} val images: {100 * correct // total} %')
    print(classification_report(val_labels, val_predicted, target_names=class_names))
    
    cm = confusion_matrix(val_labels, val_predicted)
    
    df_cm = pd.DataFrame(cm, index = class_names, columns = class_names)
    def show_confusion_matrix(confusion_matrix):
        hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        plt.ylabel("Surface Ground Truth")
        plt.xlabel("Predicted Surface")
        plt.legend()
        
    show_confusion_matrix(df_cm)


# 训练流程
def train(trainLoader, modelpath,  epoch = 150, imageSize = 224, patchSize = 14,
          numClass = 6, vitLoad = "local", freeze = False):

    # net
    dinov2_vits14 = getDinov2Vits14Model(vitLoad, imageSize, patchSize)
    modelDinov2Cls = Dinov2ClsModel(dinov2_vits14, numClass).to(device)
    modelDinov2Cls.train()

    # 损失函数的优化器定义
    criterion = nn.CrossEntropyLoss()
    optimizer = None
    if not freeze:
        optimizer = optim.Adam(modelDinov2Cls.parameters(), lr=0.000001)
    else:
        optimizer = optim.Adam(modelDinov2Cls.classifier.parameters(), lr=0.001)

    saveModelval = 0.0
    prev_params = None
    for ie in range(epoch):
        train_acc = 0
        train_loss = 0
        train_pbar = tqdm(trainLoader, desc=f"Epoch {ie+1}/{epoch} [Train]")
        num = 0

        for image, clss in train_pbar:

            image, target = image.to(device), clss.to(device)
            optimizer.zero_grad()

            preds = modelDinov2Cls(image)
            trainloss = criterion(preds, target)

            # train clss
            predictions = preds.argmax(dim=1, keepdim=True).squeeze()
            correct = (predictions == target).sum().item()
            if(len(predictions.shape) > 0):
                accuracy = correct / predictions.shape[0]
            else:
                accuracy = 0.0

            trainloss.backward()
            optimizer.step()

            num+=1.0
            train_acc += correct
            train_loss += trainloss
            train_pbar.set_postfix(loss=f"{trainloss.item():.4f}", acc=f"{accuracy:.4f}")

        train_loss /= num
        train_acc /= num
        if (saveModelval < (train_acc + (1- train_loss))) :
            torch.save(modelDinov2Cls, modelpath)

        current_params = {name: param.data.clone() for name, param in modelDinov2Cls.named_parameters() if param.requires_grad}
    
        if prev_params is not None:
            changed = compare_state_dicts(prev_params, current_params, threshold=1e-6)
            print(f"Epoch {epoch}: {len(changed)} 层参数更新")
            print(changed)
        
        prev_params = current_params
        


def predict(imagepath, modelpath, inputSizeW, inputSizeH, patchSize):
    
    modelDinov2Cls = torch.load(modelpath).to(device)
    modelDinov2Cls.eval()
    
    with torch.no_grad():
        image = cv2.imread(imagepath)
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 先试试简单的 resize 
        image = cv2.resize(imageRGB, (inputSizeW, inputSizeH), interpolation=cv2.INTER_LINEAR)
        
        # 4. 转换为 float32 并缩放到 [0, 1]（等效 ToTensor）
        img_float = image.astype(np.float32) / 255.0

        # 5. 归一化 (Normalize)
        # 将 mean 和 std 转换为 numpy 数组并 reshape 以适配广播
        mean_arr = np.array(PIXEL_MEANS, dtype=np.float32).reshape(1, 1, 3)
        std_arr = np.array(PIXEL_STDS, dtype=np.float32).reshape(1, 1, 3)
        img_norm = (img_float - mean_arr) / std_arr

        # 6. 将 HWC 转换为 CHW（PyTorch 张量格式）
        img_chw = np.transpose(img_norm, (2, 0, 1))
        imageTensor = torch.from_numpy(img_chw).unsqueeze(dim=0).to(device)

        preds = modelDinov2Cls(imageTensor)
        predictions = preds.argmax(dim=1, keepdim=True).squeeze()

        print(preds)

if __name__ == "__main__":

    BatchSize = 32
    epoch = 2
    imageSize = 224
    patchSize = 14
    numClass = 5
    class_names = ['calss_0', 'calss_1', 'calss_2', 'calss_3', 'calss_4']#['calss_0', 'calss_1', 'calss_2', 'calss_3', 'calss_4', 'calss_5']
    modelpath = "best_cls_finetune.pth"
    vitLoad = "local"
    freeze = True
    dataTrans = "myself" # transform myself
    dataDir = "/home/hualulu/code/dinov2/data/battery/"
    
    trainLoader, valLoader = getDinov2ClsDatasets(dataDir, dataTrans, imageSize, BatchSize, PIXEL_MEANS, PIXEL_STDS)
    train(trainLoader, modelpath,  epoch, imageSize, patchSize, numClass, vitLoad, freeze)
    test(valLoader, modelpath, class_names)

    
    #imagepath ="/home/hualulu/code/dinov2/data/battery/val/calss_4/image_7334_06.jpg"
    #predict(imagepath, modelpath, imageSize, imageSize, patchSize)
    
