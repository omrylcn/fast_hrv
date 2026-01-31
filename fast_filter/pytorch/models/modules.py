"""
DeepFilter - Filter Modules (Yapı Taşları)

Modellerin içinde tekrar tekrar kullanılan bloklar:
- LFilterModule: Linear (aktivasyonsuz) multi-kernel convolution
- NLFilterModule: Non-linear (ReLU) multi-kernel convolution
- LANLFilterModule: Linear + Non-Linear birleşimi
- LANLFilterDilatedModule: Dilated (genişletilmiş) versiyon
"""

import torch
import torch.nn as nn


class LFilterModule(nn.Module):
    """
    Linear Filter Module - 4 paralel Conv1D dalı

    Yapı:
        Input ─┬─ Conv1D(k=3) ─┬─ Concat ─ Output
               ├─ Conv1D(k=5) ─┤
               ├─ Conv1D(k=9) ─┤
               └─ Conv1D(k=15)─┘

    Aktivasyon: YOK (linear) - negatif değerler korunur
    Her dal: out_channels // 4 kanal üretir

    Args:
        in_channels: Giriş kanal sayısı
        out_channels: Çıkış kanal sayısı (4'e bölünmeli)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        branch_ch = out_channels // 4  # Her dal için kanal sayısı

        # 4 farklı kernel boyutu
        # padding = kernel_size // 2 -> output boyutu = input boyutu
        self.conv3 = nn.Conv1d(in_channels, branch_ch, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, branch_ch, kernel_size=5, padding=2)
        self.conv9 = nn.Conv1d(in_channels, branch_ch, kernel_size=9, padding=4)
        self.conv15 = nn.Conv1d(in_channels, branch_ch, kernel_size=15, padding=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, sequence_length)
        Returns:
            (batch, out_channels, sequence_length)
        """
        b0 = self.conv3(x)
        b1 = self.conv5(x)
        b2 = self.conv9(x)
        b3 = self.conv15(x)

        return torch.cat([b0, b1, b2, b3], dim=1)
