"""Fixed signed spectral substitution followed by a trainable Conv--BN--SiLU."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class _ConvSiBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, 1, bias=False)
        self.bn = nn.BatchNorm2d(3, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SPSM(nn.Module):
    def __init__(self, c1=3, c2=3, prior_path=None):
        super().__init__()
        if c1 != 3 or c2 != 3:
            raise ValueError("SPSM expects a three-channel detector input and output")
        prior_path = prior_path or os.environ.get("SPSM_PRIOR_PATH")
        if not prior_path:
            raise ValueError("Supply prior_path or set SPSM_PRIOR_PATH")
        prior = np.load(Path(prior_path), allow_pickle=False)
        if prior.shape != (640, 640) or not np.isfinite(prior).all():
            raise ValueError("The paper prior must be a finite 640x640 array")
        self.cv2 = _ConvSiBN()
        self.register_buffer("magnitude", torch.tensor(prior, dtype=torch.float32)[None, None], persistent=True)
        self.magnitude_eval = None

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.magnitude_eval = None
        return result

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.magnitude_eval = None

    @staticmethod
    def _rgb2gray(x):
        if x.shape[1] == 1:
            return x
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def forward(self, x):
        gray = self._rgb2gray(x)
        spectrum = torch.fft.fftshift(torch.fft.fft2(gray.float()))
        phase = spectrum.angle()
        shape = phase.shape[-2:]
        if self.training and shape == self.magnitude.shape[-2:]:
            prior = self.magnitude
            self.magnitude_eval = None
        else:
            if self.magnitude_eval is None or self.magnitude_eval.shape[-2:] != shape:
                pad_h = int((shape[0] - self.magnitude.shape[-2]) / 2)
                pad_w = int((shape[1] - self.magnitude.shape[-1]) / 2)
                if pad_h != 0 or pad_w != 0:
                    self.magnitude_eval = F.pad(self.magnitude, [pad_w, pad_w, pad_h, pad_h], "constant", 0)
                else:
                    self.magnitude_eval = self.magnitude.clone()
            if self.magnitude_eval.device != phase.device:
                self.magnitude_eval = self.magnitude_eval.to(phase.device)
            prior = self.magnitude_eval
        substituted = prior * torch.exp(1j * phase)
        reconstructed = torch.fft.ifft2(torch.fft.ifftshift(substituted), dim=(-2, -1)).real.to(dtype=x.dtype)
        return self.cv2(reconstructed)
