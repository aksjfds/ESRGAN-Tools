import cv2
import os
from ESRGAN_Tools.LG_dataset import Blur, Noise, JPEGCompression

# 视频路径
video_path = '/kaggle/input/videos/04.mp4'
# 保存图片的文件夹路径
output_dir = '/kaggle/working/data/dataLG'
os.makedirs(output_dir, exist_ok=True)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率和总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 计算间隔：如果总帧数小于1024帧，就保存所有帧
interval = max(1, total_frames // 1024)

# 抽取帧
frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 按间隔保存帧
    if frame_count % 24 == 0 and saved_count < 1024:
        # 保存帧
        output_path = os.path.join(output_dir, f'frame_{saved_count:04d}.jpg')
        
        frame = Blur(frame)
        frame = Noise(frame)
        frame = JPEGCompression(frame)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        cv2.imwrite(output_path, frame)
        saved_count += 1
    
    frame_count += 1

cap.release()
print(f"Saved {saved_count} frames to {output_dir}")


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义图像转换
transform = transforms.Compose([
#     transforms.Resize((1280, 720)),  # 如果需要调整尺寸
    transforms.ToTensor(),           # 转换为Tensor并归一化到[0, 1]
])

# 加载图像数据集
dataset = datasets.ImageFolder(root='/kaggle/working/data', transform=transform)

# 创建DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)



# 测试DataLoader
for images in dataloader:
    print(images[0].size())
    # 进一步的操作可以在这里进行
#     break  # 仅打印一次以验证输出