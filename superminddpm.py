"""
Extremely Minimalistic Implementation of DDPM

https://arxiv.org/abs/2006.11239

Everything is self contained. (Except for pytorch and torchvision... of course)

run it with `python superminddpm.py`
"""

from typing import Dict, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import time


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 3, padding=1),
    nn.BatchNorm2d(oc),
    nn.LeakyReLU(),
)
class DummyEpsModel(nn.Module):
    def __init__(self, n_channel: int) -> None:
        super(DummyEpsModel, self).__init__()
        self.conv = nn.Sequential(  # with batchnorm
            # # down sampling
            # blk(n_channel, 3),
            # blk(3, 8),
            # blk(8, 16),

            # # # up sampling
            # blk(16, 8),
            # blk(8, 3),
            # nn.Conv2d(3, n_channel, 3, padding=1),

            # old
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
class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
            x.device
        )  # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(x_i, i / self.n_T)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i


class Half:
    def __call__(self, image):
        # 將 PIL 圖像轉換為張量
        image_tensor = transforms.ToTensor()(image)
        
        # half size: MNIST->14, CIFAR10->16
        half = image_tensor[:, 14:, :] 
        # print(right_half.size())
        return half


def train_mnist(n_epoch: int = 200, device="cuda:0") -> None:

    ddpm = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)
    ddpm.to(device)
    
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

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=20)
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

    start_time = time.time()
    for i in range(n_epoch):
        ddpm.train()
        # print(dataloader)

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
            xh = ddpm.sample(4, (1, 14, 28), device)
            # xh = ddpm.sample(4, (3, 16, 32), device)
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"./contents/ddpm_sample_{i}.png")

            # save model
            # torch.save(ddpm.state_dict(), f"./ddpm_cifar.pth")
            torch.save(ddpm.state_dict(), f"./ddpm_fmnist_100_32.pth")

    stop_time = time.time()
    print(f"Total time: {stop_time - start_time}")


if __name__ == "__main__":
    train_mnist()
