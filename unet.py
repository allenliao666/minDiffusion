

from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid



blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 3, padding=1),
    nn.BatchNorm2d(oc),
    nn.LeakyReLU(),
)
class DummyEpsModel(nn.Module):
    def __init__(self, n_channel: int) -> None:
        super(DummyEpsModel, self).__init__()
        self.conv = nn.Sequential(  # with batchnorm
            # down sampling
            blk(n_channel, 32),
            blk(32, 64),
            blk(64, 128),
            blk(128, 256),

            # up sampling
            blk(256, 128),
            blk(128, 64),
            blk(64, 32),
            nn.Conv2d(32, n_channel, 3, padding=1),
        )

    def forward(self, x, t) -> torch.Tensor:
        # Lets think about using t later. In the paper, they used Tr-like positional embeddings.
        return self.conv(x)
    

class Unet(nn.Module):
    def __init__(self, n_channel: int) -> None:
        super(Unet, self).__init__()

        self.conv = nn.Sequential(  # with batchnorm
            # down sampling
            blk(n_channel, 32),
            blk(32, 64),
            blk(64, 128),

            # up sampling
            blk(128, 64),
            blk(64, 32),
            nn.Conv2d(32, n_channel, 3, padding=1),
        )
 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        x_i = torch.randn(n_sample, *size).to(device) 
        return x_i


class Half:
    def __call__(self, image):
        # 將 PIL 圖像轉換為張量
        image_tensor = transforms.ToTensor()(image)
        half = image_tensor[:, 14:, :] 
        # print(right_half.size())
        return half


def train_mnist(n_epoch: int = 200, device="cuda:0") -> None:
    unet = Unet(1)
    unet.to(device)

    tf = transforms.Compose(
        [Half(), transforms.Normalize((0.5,), (1.0))]
    )

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=20)
    optim = torch.optim.Adam(unet.parameters(), lr=2e-4)
    criterion = nn.MSELoss()  

    for i in range(n_epoch):
        unet.train()
        # print(dataloader)

        pbar = tqdm(dataloader)
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            x_bar = unet(x)
            loss = criterion(x, x_bar)
            loss.backward()
            pbar.set_description(f"loss: {loss:.4f}")
            optim.step()

        unet.eval()
        with torch.no_grad():
            xh = unet.sample(4, (1, 14, 28), device)
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"./contents/unet_sample_{i}.png")

            # save model
            torch.save(unet.state_dict(), f"./unet_mnist.pth")

if __name__ == "__main__":
    train_mnist()


