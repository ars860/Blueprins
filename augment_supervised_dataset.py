import argparse
import random
import zipfile
from os.path import basename, splitext
from pathlib import Path
from tempfile import TemporaryFile

import shutil
import numpy as np
from PIL import Image

from dataset import get_dataloaders_supervised, BlueprintsSupervisedDataset


# from albumentations.augmentations.functional import hflip, vflip, cutout


# from albumentations.augmentations.transforms import CoarseDropout


def cutout_augmentation(img, mask, min_patch_size=0, max_patch_size=50, patches_cnt=10, color=1.0,
                        remove_from_mask=True):
    img = img.squeeze()
    w, h = img.shape

    for _ in range(patches_cnt):
        x1 = random.randrange(w - min_patch_size)
        x2 = x1 + random.randrange(min_patch_size, min(max_patch_size, w - x1))
        y1 = random.randrange(h - min_patch_size)
        y2 = y1 + random.randrange(min_patch_size, min(max_patch_size, h - y1))

        img[x1:x2, y1:y2] = color
        if remove_from_mask:
            mask[:, x1:x2, y1:y2] = 0

    return img, mask


def hflip(img):
    return np.ascontiguousarray(img[:, ::-1, ...])


def vflip(img):
    return np.ascontiguousarray(img[:, :, ::-1, ...])


def flips_augmentation(img, mask, v=True, h=True):
    flips = [lambda i, m: (i, m)]
    if v:
        flips.append(lambda i, m: (vflip(i), vflip(m)))
    if h:
        flips.append(lambda i, m: (hflip(i), hflip(m)))

    results = []
    for flip in flips:
        results.append(flip(img, mask))

        # results.append(cutout_augmentation(img_a, mask_a))

    return results


def times(augmentation, cnt):
    def res(img, mask):
        results = []
        for _ in range(cnt):
            results.append(augmentation(img, mask))

        return results

    return res


def chain(*augmentations):
    def res(img, mask):
        results = [(img, mask)]

        for aug in augmentations:
            new_results = []

            for img, mask in results:
                new_results += aug(img, mask)

            results = new_results

        return results

    return res


def get_augmentation(args):
    cutout = lambda img, mask: cutout_augmentation(img.copy(), mask.copy(),
                                                   patches_cnt=args.cnt,
                                                   min_patch_size=args.min_size,
                                                   max_patch_size=args.max_size,
                                                   color=args.val,
                                                   remove_from_mask=args.cut_mask)
    cutout = times(cutout, args.times)
    flips = lambda img, mask: flips_augmentation(img, mask, args.v, args.h)

    return chain(flips, cutout)


def augment_dataset(dataset_train: BlueprintsSupervisedDataset, dataset_test: BlueprintsSupervisedDataset, args):
    if (Path() / args.root).exists():
        shutil.rmtree(Path() / args.root)

    augmentation = get_augmentation(args)

    # (Path() / args.root / args.projs).rmdir()
    (Path() / args.root / 'train' / args.projs).mkdir(parents=True, exist_ok=True)
    (Path() / args.root / 'test' / args.projs).mkdir(parents=True, exist_ok=True)
    args.max_size = max(args.max_size, args.min_size) + 1

    train_path = Path() / args.root / 'train'
    test_path = Path() / args.root / 'test'

    print('Processing train:')
    with zipfile.ZipFile(train_path / args.masks, 'w', zipfile.ZIP_DEFLATED) as mask_file:
        for i, ((img, mask), img_name, mask_name) in enumerate(
                zip(dataset_train, dataset_train.image_names, dataset_train.mask_names)):
            img, mask = img.numpy(), mask.numpy()
            img_name, img_ext = splitext(basename(img_name))
            # mask_name, mask_ext = splitext(mask_name)
            mask_ext = '.npy'

            if not args.drop_initial:
                Image.fromarray(np.uint8(img.squeeze() * 255), 'L').save(
                    train_path / args.projs / f'{img_name}{img_ext}')

                with TemporaryFile() as numpy_temp:
                    np.save(numpy_temp, mask)
                    numpy_temp.seek(0)
                    mask_file.writestr(f'{mask_name}{mask_ext}', numpy_temp.read())

            augmented = augmentation(img, mask)

            for j, (img_aug, mask_aug) in enumerate(augmented):
                Image.fromarray(np.uint8(img_aug.squeeze() * 255), 'L').save(
                    train_path / args.projs / f'{img_name}_{j}{img_ext}')

                with TemporaryFile() as numpy_temp:
                    np.save(numpy_temp, mask_aug)
                    numpy_temp.seek(0)
                    mask_file.writestr(f'{mask_name}_{j}{mask_ext}', numpy_temp.read())

            if args.test:
                break

            if i % 10 == 0:
                print(f'Files processed: {i}/{len(dataset_train)}')

    print('Processing test:')
    with zipfile.ZipFile(test_path / args.masks, 'w', zipfile.ZIP_DEFLATED) as mask_file:
        for i, ((img, mask), img_name, mask_name) in enumerate(
                zip(dataset_test, dataset_test.image_names, dataset_test.mask_names)):
            img, mask = img.numpy(), mask.numpy()
            img_name, img_ext = splitext(basename(img_name))
            mask_ext = '.npy'

            Image.fromarray(np.uint8(img.squeeze() * 255), 'L').save(
                test_path / args.projs / f'{img_name}{img_ext}')

            with TemporaryFile() as numpy_temp:
                np.save(numpy_temp, mask)
                numpy_temp.seek(0)
                mask_file.writestr(f'{mask_name}{mask_ext}', numpy_temp.read())

            if args.test:
                break

            if i % 10 == 0:
                print(f'Files processed: {i}/{len(dataset_test)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='cutout')
    parser.add_argument('--cnt', type=int, default=50)
    parser.add_argument('--times', type=int, default=2)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max_size', type=int, default=50)
    parser.add_argument('--min_size', type=int, default=0)
    parser.add_argument('--val', type=float, default=1.0)

    parser.add_argument('--root', type=str, default='blueprints')
    parser.add_argument('--masks', type=str, default='mask.zip')
    parser.add_argument('--projs', type=str, default='projs')
    # parser.add_argument('--postfix', type=str, default='projs_cutout')

    parser.add_argument('--drop_initial', action='store_true')
    parser.add_argument('--cut_mask', action='store_true')
    parser.add_argument('--test', action='store_true')

    parser.add_argument('--v', action='store_true')
    parser.add_argument('--h', action='store_true')

    parser.add_argument('--shuffle', type=int, default=None)
    args = parser.parse_args()

    if args.root == 'blueprints' and (args.masks == 'mask.zip' or args.projs == 'projs'):
        raise ValueError("Ne lez' debil!")

    if args.seed is not None:
        random.seed(args.seed)

    dataset_train, _, dataset_test, _ = get_dataloaders_supervised(shuffle_seed=args.shuffle)

    # samples = list(itertools.islice((iter(dataloader)), 1))
    #
    # image, mask = samples[0]
    # Image.fromarray(np.uint8(image.numpy().squeeze() * 255), 'L').show()

    augment_dataset(dataset_train, dataset_test, args)
    # cutout_augmentation(samples[0][0], samples[0][1])
