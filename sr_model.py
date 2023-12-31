import torch
import torch.nn as nn
import torchvision.models as models

class CNNSuperResolutionModel(nn.Module):
    def __init__(self):
        super(CNNSuperResolutionModel, self).__init__()
        self.layers1 = nn.Sequential(
            #Starting Block
            nn.Conv2d(1,64, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(1)
        )

    def forward(self, X:torch.Tensor) -> torch.Tensor:
            y = self.layers1(X)
            return y

        
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features[:8].cuda().eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
            
    def single_to_three_channels(self, x):
        return x.repeat(1, 3, 1, 1)

    def forward(self, x, y):
        x, y = self.single_to_three_channels(x), self.single_to_three_channels(x)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = nn.functional.mse_loss(x_vgg, y_vgg) + 1 * nn.functional.mse_loss(x,y) # UPDATED LOSS
        return loss