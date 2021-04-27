import itertools
import os.path
import random
from pathlib import Path

from PIL import Image
from tempfile import TemporaryFile
import zipfile
from os.path import basename, splitext

import numpy as np

from dataset import get_dataloaders_supervised, BlueprintsSupervisedDataset


def cutout_augmentation(img, mask, max_patch_size=50, patches_cnt=10):
    img = img.squeeze()
    w, h = img.shape

    for _ in range(patches_cnt):
        x1 = random.randrange(w)
        x2 = x1 + random.randrange(min(max_patch_size, w - x1))
        y1 = random.randrange(h)
        y2 = y1 + random.randrange(min(max_patch_size, h - y1))

        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        img[x1:x2, y1:y2] = 1
        mask[x1:x2, y1:y2] = 0

    img, mask = img[np.newaxis, np.newaxis, ...], mask[np.newaxis, np.newaxis, ...]

    return img, mask


def augment_dataset_cutout(dataset: BlueprintsSupervisedDataset, postfix='cutout', times=2):
    with zipfile.ZipFile(Path() / 'blueprints' / 'mask_cutout.zip', 'w') as mask_file:
        for (img, mask), img_name, mask_name in zip(dataset, dataset.image_names, dataset.mask_names):
            img, mask = img.numpy(), mask.numpy()
            img_name, img_ext = splitext(basename(img_name))
            # mask_name, mask_ext = splitext(mask_name)
            mask_ext = '.npy'

            Image.fromarray(np.uint8(img.squeeze() * 255), 'L').save(
                Path() / 'blueprints' / 'projs_cutout' / f'{img_name}{img_ext}')

            with TemporaryFile() as numpy_temp:
                np.save(numpy_temp, mask)
                mask_file.writestr(f'{mask_name}{mask_ext}', numpy_temp.read())

            for i in range(times):
                img_cutout, mask_cutout = cutout_augmentation(img.copy(), mask.copy(), patches_cnt=20)

                Image.fromarray(np.uint8(img_cutout.squeeze() * 255), 'L').save(
                    Path() / 'blueprints' / 'projs_cutout' / f'{img_name}_{postfix}_{i}{img_ext}')

                with TemporaryFile() as numpy_temp:
                    np.save(numpy_temp, mask_cutout)
                    mask_file.writestr(f'{mask_name}_{postfix}_{i}{mask_ext}', numpy_temp.read())

            # break


if __name__ == '__main__':
    dataset, dataloader = get_dataloaders_supervised(fraction=1.0)

    # samples = list(itertools.islice((iter(dataloader)), 1))
    #
    # image, mask = samples[0]
    # Image.fromarray(np.uint8(image.numpy().squeeze() * 255), 'L').show()

    augment_dataset_cutout(dataset)
    # cutout_augmentation(samples[0][0], samples[0][1])
