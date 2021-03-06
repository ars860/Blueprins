import os
import os.path as path
import re
from functools import reduce
from pathlib import Path
from typing import Any, Callable

import albumentations as A
import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
import torchvision.transforms as tr
from PIL import Image
from albumentations.pytorch import ToTensorV2
from pdf2image import convert_from_path
from torch.utils.data import Dataset, DataLoader


def filter_cutout(names):
    return list(filter(lambda name: 'cutout' not in path.basename(str(name)), names))


class BlueprintsSupervisedDataset(Dataset):
    def __init__(self,
                 root: str,
                 image_folder: str,
                 mask_folder: str,
                 transforms: [Callable] = A.Compose([A.SmallestMaxSize(256), ToTensorV2()]),
                 seed: int = None,
                 fraction: float = 0.9,
                 mode: str = None,
                 zip_archive=True,
                 filter_test=None,
                 channels=None) -> None:
        if transforms is None:
            transforms = A.Compose([A.SmallestMaxSize(256), ToTensorV2()])

        self.channels = channels
        self.root = root
        self.transforms = transforms
        self.zip_archive = zip_archive
        self.filter_test = filter_test
        self.mode = mode

        self.image_folder_path = Path(self.root) / image_folder
        self.mask_folder_path = Path(self.root) / mask_folder
        if not self.image_folder_path.exists():
            raise OSError(f"{self.image_folder_path} does not exist.")
        if not self.mask_folder_path.exists():
            raise OSError(f"{self.mask_folder_path} does not exist.")

        self.image_names = np.array(sorted(self.image_folder_path.glob("*")))
        if not zip_archive:
            self.mask_names = np.array(sorted(self.mask_folder_path.glob("*")))
        else:
            self.mask_names = np.array(list(
                map(lambda name: 'mask/' + path.splitext(path.split(name)[1])[0], self.image_names)))

        if fraction:
            if mode not in ["train", "test"]:
                raise (ValueError(
                    f"{mode} is not a valid input. Acceptable values are Train and Test."
                ))
            self.fraction = fraction

            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_names))
                np.random.shuffle(indices)
                self.image_names = self.image_names[indices]
                self.mask_names = self.mask_names[indices]

            if mode == "train":
                self.image_names = self.image_names[:int(
                    np.ceil(len(self.image_names) * self.fraction))]
                self.mask_names = self.mask_names[:int(
                    np.ceil(len(self.mask_names) * self.fraction))]
            else:
                self.image_names = self.image_names[
                                   int(np.ceil(len(self.image_names) * self.fraction)):]
                self.mask_names = self.mask_names[
                                  int(np.ceil(len(self.mask_names) * self.fraction)):]

                if self.filter_test is not None:
                    self.image_names, self.mask_names = self.filter_test(self.image_names), self.filter_test(
                        self.mask_names)

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        with open(image_path, "rb") as image_file:
            image = np.array(Image.open(image_file))  # 255 - np.array(Image.open(image_file))

            if self.zip_archive:
                with np.load(self.mask_folder_path) as archive:
                    mask = archive[mask_path]
            else:
                with open(mask_path, "rb") as mask_file:
                    mask = np.load(mask_file)

            # mask = np.moveaxis(mask, 0, 2)
            # mask = np.amax(mask, axis=2)
            mask[mask != 0] = 1

            if self.transforms and self.mode == 'train':
                # image = self.transforms(image)
                # mask = self.transforms(mask)

                augmented = self.transforms(image=image/255, masks=[m for m in mask])
            else:
                augmented = A.Compose([A.SmallestMaxSize(256), ToTensorV2()])(image=image/255, masks=[m for m in mask])

            image, mask = augmented['image'].float(), torch.FloatTensor(np.stack(augmented['masks']))

            if self.channels is not None:
                return self.channels, image, mask

            return image, mask


def get_dataloaders_supervised(root='blueprints', image_folder='projs', mask_folder='mask.zip',
                               batch_size=1,  # images have different size
                               workers=2,
                               fraction=0.9,
                               filter_test=False,
                               shuffle_seed=None,
                               transforms=ToTensorV2()):
    glob = list(map(os.path.basename, list((Path() / root).glob('*'))))

    if 'train' in glob and 'test' in glob:
        dataset_train = BlueprintsSupervisedDataset(str(Path() / root / 'train'), image_folder, mask_folder,
                                                    mode='train', fraction=None, transforms=transforms)
        dataset_test = BlueprintsSupervisedDataset(str(Path() / root / 'test'), image_folder, mask_folder, mode='test',
                                                   fraction=None, transforms=transforms)

        return dataset_train, DataLoader(dataset_train, batch_size=batch_size, num_workers=workers), \
               dataset_test, DataLoader(dataset_test, batch_size=batch_size, num_workers=workers)

    dataset_train = BlueprintsSupervisedDataset(root, image_folder, mask_folder, mode='train', fraction=fraction,
                                                seed=shuffle_seed, transforms=transforms)

    if fraction == 1.0:
        return dataset_train, DataLoader(dataset_train, batch_size=batch_size, num_workers=workers)

    dataset_test = BlueprintsSupervisedDataset(root, image_folder, mask_folder, mode='test', fraction=fraction,
                                               filter_test=filter_cutout if filter_test else None, seed=shuffle_seed, transforms=transforms)

    return dataset_train, DataLoader(dataset_train, batch_size=batch_size, num_workers=workers), \
           dataset_test, DataLoader(dataset_test, batch_size=batch_size, num_workers=workers)


def filter_files(file_names, clear=False):
    extensions = ['.pdf']
    SB_suffixes = ['????', '????']

    def is_SB(fn):
        # _, file_extension = path.splitext(fn)
        _, filename = path.split(fn)
        return reduce(lambda x, y: x or y, [suffix in filename for suffix in SB_suffixes])

    def get_SB_name(fn):
        _, filename = path.split(fn)

        exceptions = ['??????????']
        if reduce(lambda x, y: x or y, [exc in filename for exc in exceptions]):
            return filename

        return re.match(r"^([\d._]+)\s*((????)|(????))?\s*[^.]*\.\w*$", filename).groups()[0]

        # for suffix in SB_suffixes:
        #     if suffix in filename:
        #         return filename.replace(suffix, '')
        #
        # return filename

    images = list(filter(lambda s: path.splitext(s)[1] in extensions, file_names))
    SB_names = list(map(get_SB_name, filter(is_SB, file_names)))
    SB_names = SB_names + ['150', '160', '200', '120', '330', '410', '430', '4440.17.00.020', '30_120',
                           '4440.40.30.070']

    filtered_files = np.array(list(filter(lambda fn: get_SB_name(fn) not in SB_names, images)))

    if clear:
        for file_name in file_names:
            if file_name not in filtered_files:
                os.remove(file_name)
                print(f"deleted {file_name}")

    return filtered_files


class BlueprintsUnsupervisedDataset(Dataset):
    def __init__(self,
                 root: str,
                 image_folder: str,
                 dpi=200,
                 transforms: [Callable] = None,
                 seed: int = None,
                 fraction: float = 0.9,
                 mode: str = None,
                 file_format=None,
                 filter_required=False) -> None:
        if file_format is not None and not (file_format == 'pdf' or file_format == 'pil'):
            raise ValueError('Supported formats: "pdf", "pil"')
        self.file_format = file_format

        if transforms is None:
            transforms = tr.ToTensor()
        # else:
        #     transforms = tr.Compose([tr.ToTensor(), transforms])

        self.root = root
        self.transforms = transforms
        self.dpi = dpi

        image_folder_path = Path(self.root) / image_folder
        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")

        if not fraction:
            self.image_names = sorted(image_folder_path.glob("*"))

            if filter_required:
                self.image_names = filter_files(self.image_names)
        else:
            if mode not in ["train", "test"]:
                raise (ValueError(
                    f"{mode} is not a valid input. Acceptable values are train and test."
                ))
            self.fraction = fraction
            self.image_list = np.array(sorted(image_folder_path.glob("*")))

            if filter_required:
                self.image_names = filter_files(self.image_names)

            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]

            if mode == "train":
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list) * self.fraction))]
            else:
                self.image_names = self.image_list[
                                   int(np.ceil(len(self.image_list) * self.fraction)):]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]
        # print(image_path)

        # with open(image_path, "rb") as image_file:
        if self.file_format is None:
            _, file_extension = path.splitext(image_path)
            if file_extension == '.pdf':
                self.file_format = "pdf"
            else:
                self.file_format = "pil"

        if self.file_format == 'pdf':
            image = convert_from_path(image_path, dpi=self.dpi)[0].convert('L')  # Image.open(image_file)
        else:
            image = Image.open(image_path)  # 255 - np.array(Image.open(image_path))

        if self.transforms is not None:
            image = self.transforms(image)
        return image


def get_dataloaders_unsupervised(dpi=50,
                                 root=str(Path() / 'xpc_11'),
                                 image_folder='better_projs',
                                 batch_size=1,
                                 workers=2,
                                 augmentations=None,
                                 fraction=0.9,
                                 file_format=None,
                                 shuffle_seed=None):
    dataset_train = BlueprintsUnsupervisedDataset(root, image_folder, dpi=dpi, mode='train', transforms=augmentations,
                                                  fraction=fraction, file_format=file_format, seed=shuffle_seed)

    if fraction == 1.0:
        return dataset_train, DataLoader(dataset_train, batch_size=batch_size, num_workers=workers)

    dataset_test = BlueprintsUnsupervisedDataset(root, image_folder, dpi=dpi, mode='test', transforms=augmentations,
                                                 fraction=fraction, file_format=file_format, seed=shuffle_seed)

    return dataset_train, DataLoader(dataset_train, batch_size=batch_size, num_workers=workers), \
           dataset_test, DataLoader(dataset_test, batch_size=batch_size, num_workers=workers)


def test():
    rootPath = Path() / 'xpc_11'
    dataset = BlueprintsUnsupervisedDataset(str(rootPath), '?????????????? ?????? 11', mode='train', seed=123, dpi=50)
    dataloader = DataLoader(dataset)
    for i, img in enumerate(dataloader):
        if i > 1:
            break

        img = img.squeeze().numpy()
        print(img.shape)

        image = Image.fromarray(np.uint8(img * 255))
        image.show()

        plt.imshow(img)
        # plt.title(path)
        plt.show()

# if __name__ == '__main__':
#     test()
