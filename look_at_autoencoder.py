from pathlib import Path

import numpy as np
import torch

from augmentation import AddGaussianNoise
from dataset import get_dataloaders_unsupervised
from unet import Unet

device = 'cuda'


def get_last_UpBlock_weights(model):
    _, dataloader_train, _, dataloader_test = get_dataloaders_unsupervised(dpi=50,
                                                                           workers=2,
                                                                           image_folder='projs',
                                                                           augmentations=AddGaussianNoise(std=0.5))

    values = []
    for i, (img, augmented) in enumerate(dataloader_test):
        img, augmented = img.to(device), augmented.to(device)
        skip, residual = model.get_last_block_inputs(augmented)

        values.append(torch.max(torch.abs(skip)).item() / torch.max(torch.abs(residual)).item())
        if i % 100 == 0:
            print(f'Processed: {i}/{len(dataloader_train)}')

    print(np.mean(np.array(values)))


def get_last_conv_weights(model: Unet):
    weight = model.up4.conv1.weight
    # test_mean = torch.mean(test)
    skip_part, residual_part = torch.split(weight, [weight.shape[1] // 3, weight.shape[1] // 3 * 2], dim=1)
    skip_mean, residual_mean = torch.mean(torch.abs(skip_part)).item(), torch.mean(torch.abs(residual_part)).item()

    return skip_mean, residual_mean


# skip_connection version: (NO SIGMOID)
# Если так, то со skipconection-а масимум модуля в 1.8 раз больше
# Средняя величина по части тензора соответствующей skip-у = 1e-4
# Для residual = -1e-6

# with sigmoid
# skip_mean: -0.0006291013560257852, residual_mean: -0.00014486299187410623
# all is okay
if __name__ == '__main__':
    model = Unet(layers=[8, 16, 32, 64, 128], output_channels=1).to(device)
    model.load_state_dict(
        torch.load(Path() / 'learned_models' / '03_05' / 'autoencoder' / '1e-4_50epochs.pt', map_location=device))

    # get_last_UpBlock_weights(model)

    skip_mean, residual_mean = get_last_conv_weights(model)
    print(f'skip_mean: {skip_mean}, residual_mean: {residual_mean}')
