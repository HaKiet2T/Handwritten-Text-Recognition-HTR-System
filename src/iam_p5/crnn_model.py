"""
CRNN Model - Clean Architecture
CNN -> BiLSTM -> Linear -> CTC
"""
import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    """
    Single BiLSTM layer.
    nIn -> hidden * 2 -> nOut
    """
    def __init__(self, nIn: int, nHidden: int, nOut: int):
        super().__init__()
        self.rnn    = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=False)
        self.linear = nn.Linear(nHidden * 2, nOut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, nIn)
        recurrent, _ = self.rnn(x)          # (T, B, 2*nHidden)
        T, B, H = recurrent.size()
        out = self.linear(recurrent.view(T * B, H))   # (T*B, nOut)
        return out.view(T, B, -1)           # (T, B, nOut)


class CRNN(nn.Module):
    """
    Pure CRNN for OCR:
      Input : (B, 1, 32, W)  -- grayscale, height=32
      Output: (T, B, num_classes)  -- T = W after CNN

    CNN architecture guarantees height -> 1 via two (2,1) pools:
      H=32 -> 16 -> 8 -> 4 -> 2 -> 1
    """
    def __init__(self, num_classes: int):
        super().__init__()

        # ---- CNN Backbone ----
        self.cnn = nn.Sequential(
            # Block 1: (B,1,32,W) -> (B,64,16,W)
            nn.Conv2d(1,   64,  3, 1, 1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: (B,64,16,W) -> (B,128,8,W)
            nn.Conv2d(64,  128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: (B,128,8,W) -> (B,256,4,W)
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: (B,256,4,W) -> (B,512,2,W)
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),   # pool H only

            # Block 5: (B,512,2,W) -> (B,512,1,W)
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),   # pool H only => H=1
        )

        # ---- Sequence Modeling: 2-layer BiLSTM ----
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 512),           # layer 1
            BidirectionalLSTM(512, 256, num_classes),   # layer 2 -> vocab
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)                  # (B, 512, 1, W')
        assert feat.size(2) == 1, (
            f"CNN must reduce height to 1, got {feat.size(2)}"
        )
        feat = feat.squeeze(2)              # (B, 512, W')
        feat = feat.permute(2, 0, 1)        # (T=W', B, 512)
        logits = self.rnn(feat)             # (T, B, num_classes)
        return logits
