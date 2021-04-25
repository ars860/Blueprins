import argparse

import torch.nn as nn
import torch

from augmentation import AddGaussianNoise
from dataset import get_dataloaders_unsupervised
from train import device
from unet import Unet


def train_as_autoencoder(model, data_loader, num_epochs=5, mode=None):
    if not (mode == 'train' or mode == 'test'):
        raise ValueError("mode should be 'train' or 'test'")

    model.to(device)
    if mode == 'train':
        model.train()
    else:
        model.eval()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    outputs = []
    for epoch in range(num_epochs):
        for i, img in enumerate(data_loader):
            if isinstance(img, list):
                img, augmented = img
            else:
                augmented = img

            augmented.to(device)
            img.to(device)

            optimizer.zero_grad()

            x = model(augmented)
            loss = criterion(x, img)

            if mode == 'train':
                loss.backward()
                optimizer.step()

            if i % 10 == 0 or i == len(data_loader) - 1:
                print('Epoch:{}/{}, Step:{}/{}, Loss:{:.4f}'.format(epoch + 1, num_epochs, i, len(data_loader),
                                                                    float(loss)))
            outputs.append(loss)

    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')

    model = Unet(layers=[8, 16, 32, 64, 128], output_channels=1)
    dataset_train, dataloader_train, dataset_test, dataloader_test = get_dataloaders_unsupervised(dpi=50, workers=2,
                                                                                                  augmentations=AddGaussianNoise())
    train_as_autoencoder(model, dataloader_train, mode='train', num_epochs=1)