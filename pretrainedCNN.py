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
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 3, 1, 1)   # -> [8, 14, 28]
        self.pool = nn.MaxPool2d(2, 2)         # -> [8, 7, 14]
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)  # -> [16, 7, 14]
        # 再經pool -> [16, 3, 7]
        self.fc1 = nn.Linear(16 * 3 * 7, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 3 * 7)  # 16*3*7 = 336
        x = self.fc1(x)
        return x

def train_mnist(n_epoch: int = 100, device="cuda:0") -> None:
    # 建立 CNN
    cnn = SimpleCNN(num_classes=10).to(device)

    # # mnist
    # tf = transforms.Compose(
    #     [Half(), transforms.Normalize((0.1307,), (0.3081))]
    # )
    # dataset = MNIST(
    #     "./data",
    #     train=True,
    #     download=True,
    #     transform=tf,
    # )

    # fmnist
    tf = transforms.Compose(
        [Half(), transforms.Normalize((0.2860,), (0.3530))]
    )
    dataset = FashionMNIST(
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
        torch.save(cnn.state_dict(), f"./pretrained_CNN_fmnist32.pth")
    stop_time = time.time()
    print(f"Total time: {stop_time - start_time}")

if __name__ == "__main__":
    train_mnist()