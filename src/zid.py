"""Zero-shot Image Dehazing (ZID) model adapted for underwater imagery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


@dataclass
class ZIDConfig:
    max_iterations: int = 600
    lr: float = 1e-3
    lambda_rec: float = 1.0
    lambda_atm: float = 0.1
    lambda_dark: float = 0.01
    lambda_laplacian_t: float = 0.02
    lambda_laplacian_a: float = 0.001
    device: str | None = None


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, final_activation: str = "sigmoid"):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.pool = nn.AvgPool2d(2)
        
        self.bottleneck = ResidualBlock(128)
        
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ConvBlock(128 + 64, 64)
        
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(64 + 32, 32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.final_activation = final_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        
        bn = self.bottleneck(x3)

        
        d3 = self.up3(bn)
        
        if d3.shape[2:] != x2.shape[2:]:
            d3 = F.interpolate(d3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        
        if d2.shape[2:] != x1.shape[2:]:
            d2 = F.interpolate(d2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.dec2(d2)

        out = self.out_conv(d2)

        if self.final_activation == "sigmoid":
            out = torch.sigmoid(out)
        elif self.final_activation == "tanh":
            out = torch.tanh(out)
        return out


class AtmosphericNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ResidualBlock(32),
            ConvBlock(32, 64),
            ResidualBlock(64),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return self.head(feats).view(-1, 3, 1, 1)


class ZIDModel:
    def __init__(self, config: ZIDConfig | None = None):
        self.config = config or ZIDConfig()
        device_name = self.config.device
        if device_name == "gpu":
            if torch.cuda.is_available():
                device_name = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_name = "mps"
            else:
                device_name = "cpu"
        elif not device_name:  # None or empty
            if torch.cuda.is_available():
                device_name = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_name = "mps"
            else:
                device_name = "cpu"
        self.device = torch.device(device_name)
        self.j_net = UNet(3, 3, final_activation="sigmoid").to(self.device)
        self.t_net = UNet(3, 1, final_activation="sigmoid").to(self.device)
        self.a_net = AtmosphericNet().to(self.device)

    def parameters(self):
        for net in (self.j_net, self.t_net, self.a_net):
            yield from net.parameters()

    def _estimate_prior_atmosphere(self, image: np.ndarray) -> torch.Tensor:
        flat = image.reshape(-1, 3)
        brightest_idx = np.argsort(flat.sum(axis=1))[-int(0.01 * len(flat)) :]
        estimate = flat[brightest_idx].mean(axis=0)
        return torch.from_numpy(estimate).float().to(self.device)

    def _total_variation(self, tensor: torch.Tensor) -> torch.Tensor:
        dh = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).mean()
        dw = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).mean()
        return dh + dw

    def _smooth_atmosphere(self, atmo: torch.Tensor) -> torch.Tensor:
        return (atmo - atmo.mean(dim=0, keepdim=True)).pow(2).mean()

    def _laplacian_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        
        if tensor.shape[2] < 3 or tensor.shape[3] < 3:
            return torch.tensor(0.0, device=self.device)
        
        num_channels = tensor.shape[1]
        
        kernel = torch.ones(num_channels, 1, 3, 3, device=self.device) / 9.0
        
        padded_tensor = F.pad(tensor, (1, 1, 1, 1), mode='reflect')
        
        mean_neighbors = F.conv2d(padded_tensor, kernel, groups=num_channels)
        
        return (tensor - mean_neighbors).pow(2).mean()

    def _forward_networks(self, hazy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        j_hat = self.j_net(hazy)
        t_hat = self.t_net(hazy)
        a_hat = self.a_net(hazy)
        return j_hat, t_hat, a_hat

    def _pad_to_even(self, tensor: torch.Tensor) -> Tuple[torch.ndarray, int, int]:
        _, _, h, w = tensor.shape
        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
        return tensor, h, w

    def dehaze(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hazy = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
        hazy, orig_h, orig_w = self._pad_to_even(hazy)
        hazy = hazy.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        atmosphere_prior = self._estimate_prior_atmosphere(image)

        iterator = tqdm(range(self.config.max_iterations), desc="ZID", leave=False)
        for _ in iterator:
            optimizer.zero_grad()
            j_hat, t_hat, a_hat = self._forward_networks(hazy)
            recon = j_hat * t_hat + a_hat * (1 - t_hat)
            loss_rec = F.mse_loss(recon, hazy)
            loss_atm = F.mse_loss(a_hat.view(-1, 3), atmosphere_prior.unsqueeze(0))
            dark_channel = torch.min(j_hat, dim=1)[0]
            loss_dark = dark_channel.mean()
            loss_laplacian_t = self._laplacian_norm(t_hat)
            loss_laplacian_a = self._laplacian_norm(a_hat)
            loss = (
                self.config.lambda_rec * loss_rec
                + self.config.lambda_atm * loss_atm
                + self.config.lambda_dark * loss_dark
                + self.config.lambda_laplacian_t * loss_laplacian_t
                + self.config.lambda_laplacian_a * loss_laplacian_a
            )
            loss.backward()
            optimizer.step()
            iterator.set_postfix({"L_rec": float(loss_rec.detach().cpu())})

        with torch.no_grad():
            j_hat, t_hat, _ = self._forward_networks(hazy)
        j_hat = j_hat[:, :, :orig_h, :orig_w]
        t_hat = t_hat[:, :, :orig_h, :orig_w]
        clean = j_hat.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
        transmission = t_hat.squeeze(0).squeeze(0).clamp(0.0, 1.0).cpu().numpy()
        return clean, transmission