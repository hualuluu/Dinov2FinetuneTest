# Dinov2FinetuneTest
使用Dinov2学习的一些代码记录   

## 代码使用环境： 
1. git clone https://github.com/facebookresearch/dinov2.git   
2. 按照dinov2 的requirement.txt 安装环境    
3. 分类微调：将 Dinov2ClsFinetuneTestCode.py文件放到 dinov2/主文件夹下，直接运行 python Dinov2ClsFinetuneTestCode.py即可开始训练   

## 分类微调代码：Dinov2ClsFinetuneTestCode.py   
   代码：使用Dinov2 vits14 作为backbone的分类网络简单微调训练流程   
   包括两种微调方式：仅训练分类头 和 backbone + 分类头都训练   
   如果 选择线上hub加载backbone，则不需要手动下载pretrain model， 如果选择local方式， 则要手动下载dinov2_vits14_pretrain.pth放到dinov2/主文件夹下   
