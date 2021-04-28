import argparse
import random
import zipfile
from os.path import basename, splitext
from pathlib import Path
from tempfile import TemporaryFile

import numpy as np
from PIL import Image

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

    img = img[np.newaxis, np.newaxis, ...]

    return img, mask


def augment_dataset_cutout(dataset: BlueprintsSupervisedDataset, args):
    (Path() / args.root / args.projs).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(Path() / args.root / args.masks, 'w', zipfile.ZIP_DEFLATED) as mask_file:
        for (img, mask), img_name, mask_name in zip(dataset, dataset.image_names, dataset.mask_names):
            img, mask = img.numpy(), mask.numpy()
            img_name, img_ext = splitext(basename(img_name))
            # mask_name, mask_ext = splitext(mask_name)
            mask_ext = '.npy'

            Image.fromarray(np.uint8(img.squeeze() * 255), 'L').save(
                Path() / args.root / args.projs / f'{img_name}{img_ext}')

            with TemporaryFile() as numpy_temp:
                np.save(numpy_temp, mask)
                numpy_temp.seek(0)
                mask_file.writestr(f'{mask_name}{mask_ext}', numpy_temp.read())

            for i in range(args.times):
                img_cutout, mask_cutout = cutout_augmentation(img.copy(), mask.copy(), patches_cnt=args.cnt, max_patch_size=args.max_size)

                Image.fromarray(np.uint8(img_cutout.squeeze() * 255), 'L').save(
                    Path() / args.root / args.projs / f'{img_name}_{args.type}_{i}{img_ext}')

                with TemporaryFile() as numpy_temp:
                    np.save(numpy_temp, mask_cutout)
                    numpy_temp.seek(0)
                    mask_file.writestr(f'{mask_name}_{args.type}_{i}{mask_ext}', numpy_temp.read())

            # break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='cutout')
    parser.add_argument('--cnt', type=int, default=50)
    parser.add_argument('--times', type=int, default=2)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max_size', type=int, default=50)

    parser.add_argument('--root', type=str, default='blueprints')
    parser.add_argument('--masks', type=str, default='mask_cutout.zip')
    parser.add_argument('--projs', type=str, default='projs_cutout')
    # parser.add_argument('--postfix', type=str, default='projs_cutout')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    dataset, dataloader = get_dataloaders_supervised(fraction=1.0)

    # samples = list(itertools.islice((iter(dataloader)), 1))
    #
    # image, mask = samples[0]
    # Image.fromarray(np.uint8(image.numpy().squeeze() * 255), 'L').show()

    augment_dataset_cutout(dataset, args)
    # cutout_augmentation(samples[0][0], samples[0][1])
