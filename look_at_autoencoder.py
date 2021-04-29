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
                                                                           augmentations=AddGaussianNoise(0.5))

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
    skip_mean, residual_mean = torch.mean(skip_part).item(), torch.mean(residual_part).item()

    return skip_mean, residual_mean


# skip_connection version:
# Если так, то со skipconection-а масимум модуля в 1.8 раз больше
# Средняя величина по части тензора соответствующей skip-у = 1e-4
# Для residual = -1e-6
if __name__ == '__main__':
    model = Unet(layers=[8, 16, 32, 64, 128], output_channels=1, skip=False).to(device)
    model.load_state_dict(
        torch.load(Path() / 'learned_models' / 'no_skip' / 'autoencoder' / '1e-4_10epochs.pt', map_location=device))

    # get_last_UpBlock_weights(model)
    get_last_conv_weights(model)
