import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import wandb
from albumentations.pytorch import ToTensorV2

from augmentation import AddGaussianNoise
from dataset import get_dataloaders_supervised, get_dataloaders_unsupervised
import albumentations as A

from unet import Unet

import torchvision.transforms as tr

if __name__ == '__main__':
    # device = 'cpu'
    # model = Unet(layers=[8, 16, 32, 64, 128], output_channels=11).to(device)
    # model.load_state_dict(torch.load(Path() / 'artifacts' / 'model-v321' / 'sweep_model.pt', map_location=device))
    #
    # _, dataloader_train, _, dataloader_test = get_dataloaders_supervised(transforms=A.Compose([A.SmallestMaxSize(256), A.Cutout(num_holes=100, always_apply=True), ToTensorV2()]), workers=1)
    #
    # img, mask = list(itertools.islice(iter(dataloader_train), 5))[2]
    #
    # plt.imshow(img.squeeze())
    # plt.show()
    #
    # result = model(img)

    transforms = [tr.Resize(256), tr.ToTensor(), AddGaussianNoise(std=0.5)]
    transforms = tr.Compose(transforms)

    _, dataloader_train, _, dataloader_test = get_dataloaders_unsupervised(workers=1,
                                                                           image_folder='projs',
                                                                           augmentations=transforms)

    for i, (img, augmented) in enumerate(dataloader_train):
        if i == 10:
            break

        plt.imshow(augmented.squeeze())
        plt.show()



