import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 自定义数据集类
class VideoDataset(Dataset):
    def __init__(self, video_path, transform=None, size=1024):
        self.video_path = video_path
        self.transform = transform
        self.frames = []
        self.size = size
        self._load_video()

    def _load_video(self):
        # 使用 OpenCV 读取视频
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, 15000)
        videoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        count = 0
        for i in range(int(videoFrames)):
            success, frame = cap.read()
            if i % 24 != 0:
                continue
            if not success:
                break
            
            count += 1
            if count >= self.size:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if transform:
                frame = transform(frame)
            self.frames.append(frame)
        cap.release()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

# 转换和数据加载
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量并且归一化到 [0, 1]
])

video_path = '/kaggle/input/videos/mila.mp4'
video_dataset = VideoDataset(video_path, transform=transform, size=512)

video_loader = DataLoader(video_dataset.frames, batch_size=len(video_dataset.frames), shuffle=False)

for i, video_loader in enumerate(video_loader):
    torch.save(video_loader, "mila.pt")