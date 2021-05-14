import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations import DualTransform

from dataset import get_dataloaders_unsupervised


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor, tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class HideRectangle(DualTransform):
    def __init__(self, max_cnt, channels, p=0.5, min_cnt=1, always_apply=False, fill_mask=0, fill_image=1, exclude_masks=None):
        super(HideRectangle, self).__init__(p, always_apply)
        self.min_cnt = min_cnt
        self.max_cnt = max_cnt
        self.channels = channels
        self.p = p
        self.fill_image = fill_image
        self.fill_mask = fill_mask
        if exclude_masks is None:
            exclude_masks = []
        self.exclude_masks = set(exclude_masks)

    @property
    def targets_as_params(self):
        return ["masks"]

    def get_params_dependent_on_targets(self, params):
        masks = params["masks"]

        to_erase = {}
        for channel in self.channels:
            mask_channel = masks[channel]

            mask_channel = np.array(mask_channel * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_channel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            erased_cnt = random.randint(min(self.min_cnt, self.max_cnt), self.max_cnt)
            erased_indexes = random.sample(range(len(contours)), min(erased_cnt, len(contours)))

            xy = []
            for index in erased_indexes:
                xy.append(contours[index])

            if len(xy) != 0:
                to_erase[channel] = xy

        params.update({"erase": to_erase})
        return params

    def apply(self, img, erase=None, **params):
        # print("a")
        if erase is None:
            return img

        # print(erase.items())

        # img = np.float32(img)
        for ch, things_to_erase in erase.items():
            img[cv2.drawContours(np.zeros_like(img), things_to_erase, -1, 1, thickness=cv2.FILLED) == 1] = self.fill_image

        return img

    def apply_to_masks(self, img, erase=None, **params):
        masks = img
        if erase is None:
            return masks

        # print(erase)
        for i, m in enumerate(masks):
            for ch, things_to_erase in erase.items():
                to_erase = cv2.drawContours(np.zeros_like(masks[0]), things_to_erase, -1, 1, thickness=cv2.FILLED) == 1
                if i not in self.exclude_masks:
                    m[to_erase] = self.fill_mask

        return masks


if __name__ == '__main__':
    _, dataloader_train, _, _ = get_dataloaders_unsupervised(dpi=50, workers=2, augmentations=AddGaussianNoise(std=0.5))
    img, augmented = next(iter(dataloader_train))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img.numpy().squeeze(0).transpose(1, 2, 0))
    ax2.imshow(augmented.numpy().squeeze(0).transpose(1, 2, 0))
    plt.show()
