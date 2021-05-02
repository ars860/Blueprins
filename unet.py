import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

N_CLASSES = 11


class SkipType(Enum):
    SKIP = 0
    NO_SKIP = 1
    ZERO_SKIP = 2


class UpBlock(nn.Module):
    def __init__(self, ch_out, skip=SkipType.SKIP, dropout=False):
        super(UpBlock, self).__init__()
        self.skip = skip
        self.upconv = nn.ConvTranspose2d(ch_out * 2, ch_out * 2, kernel_size=3, padding=1, stride=2, output_padding=1)
        # self.upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(ch_out * 2 if skip == SkipType.NO_SKIP else ch_out * 3, ch_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)

        if dropout:
            self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, u=None, padding=None):
        if u is None and self.skip:
            raise ValueError("Expected skip connection")

        if padding is None:
            padding = [x.shape[2] % 2, x.shape[3] % 2]

        u = self.upconv(u)
        u = F.pad(u, [padding[1], 0, padding[0], 0])

        if self.skip == SkipType.SKIP:
            x1 = torch.cat((x, u), dim=1)
        elif self.skip == SkipType.ZERO_SKIP:
            x1 = torch.cat((torch.zeros_like(x), u), dim=1)
        else:
            x1 = u

        # x1 = torch.cat((x, u), dim=1) if self.skip else u
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        if getattr(self, 'dropout', None) is not None:
            x3 = self.dropout(x3)
        return x3


class Unet(nn.Module):
    def __init__(self, layers, output_channels=N_CLASSES, skip=SkipType.SKIP, dropout=False):
        super(Unet, self).__init__()
        prev_layer = 1
        for i, layer in enumerate(layers, start=1):
            setattr(self, f'down{i}', self.enc_block(prev_layer, layer, max_pool=prev_layer != 1))
            prev_layer = layer
        for i, layer in reversed(list(enumerate(layers[:-1], start=1))):
            setattr(self, f'up{i}', UpBlock(layer, skip, dropout))
        self.final = nn.Conv2d(layers[0], output_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # l1, l2, l3, l4, l5 = layers
        # super(Unet, self).__init__()
        # self.down1 = self.enc_block(1, l1, max_pool=False)  # 256
        # self.down2 = self.enc_block(l1, l2)  # 128
        # self.down3 = self.enc_block(l2, l3)  # 64
        # self.down4 = self.enc_block(l3, l4)  # 32
        # self.down5 = self.enc_block(l4, l5)
        # self.up4 = UpBlock(l4, skip, dropout)
        # self.up3 = UpBlock(l3, skip, dropout)
        # self.up2 = UpBlock(l2, skip, dropout)
        # self.up1 = UpBlock(l1, skip, dropout)
        # self.final = nn.Conv2d(l1, output_channels, kernel_size=1)

    def forward_vars(self, x):
        children = list(self.named_children())
        vars = {}
        downs = list(map(lambda nm: nm[1], filter(lambda nm: 'down' in nm[0], children)))
        ups = list(map(lambda nm: nm[1], filter(lambda nm: 'up' in nm[0], children)))
        final = children[-2][1]
        sigmoid = children[-1][1]

        vars['x'] = x
        for i, down_i in enumerate(downs[:-1], start=1):
            vars[f'x{i}'] = down_i(vars[f'x{"" if i == 1 else i - 1}'])

        vars[f'u{len(downs)}'] = downs[-1](vars[f'x{len(downs) - 1}'])
        for i, up_i in enumerate(ups):
            vars[f'u{len(ups) - i}'] = up_i(vars[f'x{len(ups) - i}'], vars[f'u{len(ups) - i + 1}'])
        vars['res'] = final(vars['u1'])
        vars['res_sigmoid'] = sigmoid(vars['res'])
        return vars

    def forward(self, x):
        return self.forward_vars(x)['res_sigmoid']

        # x1 = self.down1(x)
        # x2 = self.down2(x1)
        # x3 = self.down3(x2)
        # x4 = self.down4(x3)
        # u5 = self.down5(x4)
        # u4 = self.up4(x4, u5)
        # u3 = self.up3(x3, u4)
        # u2 = self.up2(x2, u3)
        # u1 = self.up1(x1, u2)
        # res = self.final(u1)
        # return res

    def enc_block(self, ch_in, ch_out, max_pool=True):
        layers = [nn.MaxPool2d(kernel_size=2)] if max_pool else []
        layers += [
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        return nn.Sequential(*layers)

    def get_last_block_inputs(self, x):
        with torch.no_grad():
            vars = self.forward_vars(x)

            return vars['x1'], vars['u2']
            # x1 = self.down1(x)
            # x2 = self.down2(x1)
            # x3 = self.down3(x2)
            # x4 = self.down4(x3)
            # u5 = self.down5(x4)
            # u4 = self.up4(x4, u5, padding=[x4.shape[2] % 2, x4.shape[3] % 2])
            # u3 = self.up3(x3, u4, padding=[x3.shape[2] % 2, x3.shape[3] % 2])
            # u2 = self.up2(x2, u3, padding=[x2.shape[2] % 2, x2.shape[3] % 2])
            #
            # return x1, u2

    # def sigmoid(self, x):
    #     with torch.no_grad():
    #         return torch.sigmoid(self(x))
