from typing import Dict, Optional, Tuple
from sympy import Ci
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from mindiffusion.unet import NaiveUnet
from mindiffusion.ddpm import DDPM


class Half:
    def __call__(self, image):
        # 將 PIL 圖像轉換為張量
        image_tensor = transforms.ToTensor()(image)
        
        # half size: MNIST->14, CIFAR10->16
        half = image_tensor[:, 32:, :] 
        # print(right_half.size())
        return half

def train_tinyimagenet(
    n_epoch: int = 100, device: str = "cuda:0", load_pth: Optional[str] = None
) -> None:

    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)
    if load_pth is not None:
        ddpm.load_state_dict(torch.load("ddpm_tinyimagenet.pth"))

    ddpm.to(device)

    tf = transforms.Compose(
        [Half(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = ImageFolder(
        "~/.torchtiny-imagenet-200/train",
        transform=tf,
    )

    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=16)
    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-5)

    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"Epoch {i+1}/{n_epoch} | loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(8, (3, 16, 32), device)
            xset = torch.cat([xh, x[:8]], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            save_image(grid, f"./contents/ddpm_sample_cifar{i}.png")

            # save model
            torch.save(ddpm.state_dict(), f"./ddpm_cifar.pth")


if __name__ == "__main__":
    train_tinyimagenet()
