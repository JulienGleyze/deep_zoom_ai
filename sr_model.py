import torch
import torch.nn as nn
from math import sqrt

class CNNSuperResolutionModel(nn.Module):
    def __init__(self):
        super(CNNSuperResolutionModel, self).__init__()
        self.layers1 = nn.Sequential(
            #Starting Block
            nn.Conv2d(1,64, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(1)
        )

    def forward(self, X:torch.Tensor) -> torch.Tensor:
            y = self.layers1(X)
            return y
        
class perChanModel(nn.Module):
    def __init__(self):
        super(perChanModel, self).__init__()
        self.Rlayer = CNNSuperResolutionModel()
        self.Glayer = CNNSuperResolutionModel()
        self.Blayer = CNNSuperResolutionModel()
    
    def forward(self, X:torch.Tensor) -> torch.Tensor:
            Rx = self.Rlayer(X[:, 0].unsqueeze(1))
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            Ry = self.Glayer(X[:, 1].unsqueeze(1))
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            Rz = self.Blayer(X[:, 2].unsqueeze(1))
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            return torch.cat(Rx, Ry, Rz, dim=1)
    

class FSRCNN(nn.Module):
    """
sqr
    Args:
        upscale_factor (int): Image magnification factor.
    """

    def __init__(self, upscale_factor: int) -> None:
        super(FSRCNN, self).__init__()
        # Feature extraction layer.
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 56, (5, 5), (1, 1), (2, 2)),
            nn.PReLU(56)
        )

        # Shrinking layer.
        self.shrink = nn.Sequential(
            nn.Conv2d(56, 12, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(12)
        )

        # Mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12)
        )

        # Expanding layer.
        self.expand = nn.Sequential(
            nn.Conv2d(12, 56, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(56)
        )

        # Deconvolution layer.
        self.deconv = nn.ConvTranspose2d(56, 3, (9, 9), (upscale_factor, upscale_factor), (4, 4), (upscale_factor - 1, upscale_factor - 1))

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extraction(x)
        out = self.shrink(out)
        out = self.map(out)
        out = self.expand(out)
        out = self.deconv(out)

        return out

    # The filter weight of each layer is a Gaussian distribution with zero mean and standard deviation initialized by random extraction 0.001 (deviation is 0).
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias.data)