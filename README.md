# Dinov2FinetuneTest
使用Dinov2微调学习的一些代码记录   

## 代码使用环境： 
1. git clone https://github.com/facebookresearch/dinov2.git   
2. 按照dinov2 的requirement.txt 安装环境    
3. 分类微调：将 Dinov2ClsFinetuneTestCode.py文件放到 dinov2/主文件夹下，直接运行 python Dinov2ClsFinetuneTestCode.py即可开始训练   
4. 分割微调: 将 Dinov2SegFinetuneTestCode.py文件放到 dinov2/主文件夹下，直接运行 python Dinov2SegFinetuneTestCode.py即可开始训练 

   
## 分类微调代码：Dinov2ClsFinetuneTestCode.py   
   - 代码：使用Dinov2 vits14 作为backbone的分类网络简单微调训练流程   
   - 包括两种微调方式：仅训练分类头 和 backbone + 分类头都训练   
   - 如果 选择线上hub加载backbone，则不需要手动下载pretrain model， 如果选择local方式， 则要手动下载dinov2_vits14_pretrain.pth放到dinov2/主文件夹下      
   - 数据存放格式：      
      cls_data/   
      │   
      ├── train/   
      │   ├── class_0/   
      │   └── class_1/    
      │   
      └── val/   
      │   ├── class_0/   
      │   └── class_1/      

   
## 分割微调代码：Dinov2SegFinetuneTestCode.py   
   - 代码：使用Dinov2 vits14 作为backbone的分割网络简单微调训练流程   
   - 包括两种微调方式：仅训练分类头 和 backbone + 分类头都训练   
   - 如果 选择线上hub加载backbone，则不需要手动下载pretrain model， 如果选择local方式， 则要手动下载dinov2_vits14_pretrain.pth放到dinov2/主文件夹下
   - 包含两种分割头， head_type参数可以选择。这属于测试代码，自己可以构建其他的分割头
   - 数据存放格式：【虽然有 验证集但我实际没写验证代码，只写了一个简单的推理代码predict】      
      seg_data/   
      │   
      ├── train/   
      │   ├── image1.jpg   
      │   ├── image1.json (labelme标注的分割json格式)
      │   ├── ...
      │   
      └── val/   
      │   ├── image2.jpg   
      │   ├── image2.json (labelme标注的分割json格式)
      │   ├── ...


## 分类OnnxModel转换： 
官方dinov2 的环境为torch2.0.0,转换过程中会报错，这是因为torch本身的bug，查询git后发现后面的torch版本已经修复了，所以更新一下环境参数

```shell   
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 xformers --index-url https://download.pytorch.org/whl/cu118
pip install -e .[extras] --extra-index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.nvidia.com
```   
