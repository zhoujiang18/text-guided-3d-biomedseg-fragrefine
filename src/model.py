import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(SegmentationModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def predict(self, x, clicks=None, prev_pred=None):
        """
        Inference wrapper. Uses softmax + argmax to get segmentation.
        - x: Tensor, shape [B, C, D, H, W]
        - clicks, prev_pred: not used here but kept for compatibility
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        seg = torch.argmax(probs, dim=1)  # shape [B, D, H, W]
        return seg  # leave as tensor, conversion happens in inference.py
