"""
Simple Unet Structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )

        self.is_res = is_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main(x)
        if self.is_res:
            x = x + self.conv(x)
            return x / 1.414
        else:
            return self.conv(x)


class UnetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetDown, self).__init__()
        layers = [Conv3(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            Conv3(out_channels, out_channels),
            Conv3(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # print(f"輸入形狀: {x.size()}, 跳躍連接形狀: {skip.size()}")
        x = torch.cat((x, skip), 1)
        x = self.model(x)

        return x



class TimeSiren(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super(TimeSiren, self).__init__()

        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class NaiveUnet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_feat: int = 128) -> None:
        super(NaiveUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_feat = n_feat

        self.init_conv = Conv3(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.ReLU())

        self.timeembed = TimeSiren(2 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, kernel_size=(2,4), stride=(2,4)),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Conv2d(2 * n_feat, self.out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        x = self.init_conv(x) # [3,16,32]>[128,16,32]

        down1 = self.down1(x) # [128,16,32]>[128,8,16]
        down2 = self.down2(down1) # [128,8,16]>[256,4,8]
        down3 = self.down3(down2) # [256,4,8]>[256,2,4]

        thro = self.to_vec(down3) # [256,2,4]>[256,1,1]
        temb = self.timeembed(t).view(-1, self.n_feat * 2, 1, 1) # [256,1,1]

        thro = self.up0(thro + temb) # [512,1,1]>[512,2,4]

        up1 = self.up1(thro, down3) + temb # [512,2,4]>[512,4,8]
        up2 = self.up2(up1, down2) # [512,4,8]>[256,8,16]
        up3 = self.up3(up2, down1) # [256,8,16]>[128,16,32]

        out = self.out(torch.cat((up3, x), 1)) # [256,16,32]>[10,16,32]

        return out

    # def __init__(self, in_channels: int, out_channels: int, n_feat: int = 256) -> None:
    #     super(NaiveUnet, self).__init__()
    #     self.in_channels = in_channels
    #     self.out_channels = out_channels
    #     self.n_feat = n_feat

    #     self.init_conv = Conv3(in_channels, n_feat, is_res=True)  

    #     # Down-sampling layers
    #     self.down1 = UnetDown(n_feat, 2 * n_feat)  
    #     self.down2 = UnetDown(2 * n_feat, 2 * n_feat)  

    #     # Feature vector layer
    #     self.to_vec = nn.Sequential(nn.AvgPool2d(2), nn.ReLU())  

    #     # Time embedding
    #     self.timeembed = TimeSiren(2 * n_feat)  

    #     # Up-sampling layers
    #     self.up0 = nn.Sequential(  
    #         nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 2, 2),
    #         nn.GroupNorm(8, 2 * n_feat),
    #         nn.ReLU(),
    #     )

    #     self.up1 = UnetUp(4 * n_feat, 2 *n_feat)  
    #     self.up2 = UnetUp(4 * n_feat, n_feat)  # [256,8,16] -> [128,16,32]

    #     self.out = nn.Conv2d(2 * n_feat, self.out_channels, 3, 1, 1)  # Final output layer

    # def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    #     # Initial convolution
    #     x = self.init_conv(x) # [3,16,32] -> [128,16,32]

    #     # Down-sampling
    #     down1 = self.down1(x) # [128,16,32] -> [256,8,16]
    #     down2 = self.down2(down1) # [256,8,16] -> [256,4,8]

    #     # Feature vector
    #     thro = self.to_vec(down2) # [256,4,8] -> [256,2,4]
    #     temb = self.timeembed(t).view(-1, self.n_feat * 2, 1, 1) # [256,1,1]

    #     # Up-sampling
    #     thro = self.up0(thro + temb) # [256,2,4] -> [256,4,8]

    #     up1 = self.up1(thro, down2) + temb # [512,4,8] -> [256,8,16]
    #     up2 = self.up2(up1, down1) # [512,8,16] -> [128,16,32]

    #     # Final layer
    #     out = self.out(torch.cat((up2, x), 1)) # [128,16,32] -> [10,16,32]

    #     return out