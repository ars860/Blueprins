import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 11


class UpBlock(nn.Module):
    def __init__(self, ch_out):
        super(UpBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(ch_out * 2, ch_out * 2, kernel_size=3, padding=1, stride=2, output_padding=1)
        # self.upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # TODO: ConvTranspose2d
        self.conv1 = nn.Conv2d(ch_out * 3, ch_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)

    def forward(self, x, u, padding=None):
        if padding is None:
            padding = [x.shape[2] % 2, x.shape[3] % 2]

        u = self.upconv(u)
        u = F.pad(u, [padding[1], 0, padding[0], 0])

        diffY = u.size()[2] - x.size()[2]
        diffX = u.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        x1 = torch.cat((x, u), dim=1)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        return x3


class Unet(nn.Module):
    def __init__(self, layers, output_channels=N_CLASSES):
        # if layers is None:
        #     layers = [64, 128, 256, 512]

        l1, l2, l3, l4, l5 = layers
        super(Unet, self).__init__()
        self.down1 = self.enc_block(1, l1, max_pool=False)  # 256
        self.down2 = self.enc_block(l1, l2)  # 128
        self.down3 = self.enc_block(l2, l3)  # 64
        self.down4 = self.enc_block(l3, l4)  # 32
        self.down5 = self.enc_block(l4, l5)
        self.up4 = UpBlock(l4)
        self.up3 = UpBlock(l3)
        self.up2 = UpBlock(l2)
        self.up1 = UpBlock(l1)
        self.final = nn.Conv2d(l1, output_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        u5 = self.down5(x4)
        u4 = self.up4(x4, u5, padding=[x4.shape[2] % 2, x4.shape[3] % 2])
        u3 = self.up3(x3, u4, padding=[x3.shape[2] % 2, x3.shape[3] % 2])
        u2 = self.up2(x2, u3, padding=[x2.shape[2] % 2, x2.shape[3] % 2])
        u1 = self.up1(x1, u2, padding=[x1.shape[2] % 2, x1.shape[3] % 2])
        res = self.final(u1)
        return res

    def enc_block(self, ch_in, ch_out, max_pool=True):
        layers = [nn.MaxPool2d(kernel_size=2)] if max_pool else []
        layers += [
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        return nn.Sequential(*layers)

    def sigmoid(self, x):
        with torch.no_grad():
            return self(x)
