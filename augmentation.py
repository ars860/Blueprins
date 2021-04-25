import matplotlib.pyplot as plt
import torch

from dataset import get_dataloaders_unsupervised


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor, tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


if __name__ == '__main__':
    _, dataloader_train, _, _ = get_dataloaders_unsupervised(dpi=50, workers=2, augmentations=AddGaussianNoise())
    img, augmented = next(iter(dataloader_train))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img.numpy().squeeze(0).transpose(1, 2, 0))
    ax2.imshow(augmented.numpy().squeeze(0).transpose(1, 2, 0))
    plt.show()
