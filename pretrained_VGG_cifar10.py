from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import time



class Half:
    def __call__(self, image):
        # 將 PIL 圖像轉換為張量
        image_tensor = transforms.ToTensor()(image)
        
        # half size: MNIST->14, CIFAR10->16
        half = image_tensor[:, 14:, :] 
        return half

class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super(SimpleCNN, self).__init__()
        # act = nn.Tanh
        act = nn.ReLU
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            act(),
            nn.MaxPool2d(2, 2), #[32,16]->[16,8]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            act(),
            nn.MaxPool2d(2, 2), #[16,8]->[8,4] 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            act(),
            nn.MaxPool2d(2, 2), #[8,4]->[4,2]
        )

        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.7),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            # nn.Dropout(p=0.7),
            nn.Linear(in_features=256, out_features=n_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_mnist(n_epoch: int = 100, device="cuda:0") -> None:
    # 建立 CNN
    cnn = SimpleCNN(n_classes=10).to(device)

    # 定義 transform
    tf = transforms.Compose([
        Half(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 只是示例，實務上可用官方統計值
    ])

    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    # 使用 Adam
    optimizer = torch.optim.Adam(cnn.parameters(), lr=2e-4)
    # 損失函式：交叉熵
    criterion = nn.CrossEntropyLoss()

    # 開始訓練
    start_time = time.time()
    for epoch in range(n_epoch):
        cnn.train()
        pbar = tqdm(dataloader)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = cnn(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch+1}/{n_epoch} | loss: {loss.item():.4f}")

        # 儲存模型
        torch.save(cnn.state_dict(), f"./pretrained_LENET_cifar10_32.pth")
    stop_time = time.time()
    print(f"Total time: {stop_time - start_time}")

if __name__ == "__main__":
    train_mnist()