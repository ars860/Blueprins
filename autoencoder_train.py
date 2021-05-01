import argparse
import os
from pathlib import Path

import numpy as np
import torch.nn as nn
import torch

from augmentation import AddGaussianNoise
from dataset import get_dataloaders_unsupervised
from train_segmentation import device
from unet import Unet


def train_as_autoencoder(model, data_loader, test_loader, num_epochs=5, mode=None, device=device, lr=1e-3, invert=False):
    if not (mode == 'train' or mode == 'test'):
        raise ValueError("mode should be 'train' or 'test'")

    model = model.to(device)
    if mode == 'train':
        model.train()
    else:
        model.eval()

    criterion = nn.MSELoss()  # ???
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    outputs = []
    for epoch in range(num_epochs):
        train_losses = np.zeros(len(data_loader))

        for i, img in enumerate(data_loader):
            if isinstance(img, list):
                img, augmented = img
            else:
                augmented = img

            augmented, img = augmented.to(device), img.to(device)

            optimizer.zero_grad()

            x = model(augmented)
            x = torch.sigmoid(x)
            loss = criterion(x, img if not invert else 1. - img)

            if mode == 'train':
                loss.backward()
                optimizer.step()

            train_losses[i] = loss.item()
            if i % 10 == 0 or i == len(data_loader) - 1:
                print('Epoch:{}/{}, Step:{}/{}, Loss:{:.4f}'.format(epoch + 1, num_epochs, i, len(data_loader),
                                                                    np.true_divide(train_losses.sum(),
                                                                                   (train_losses != 0).sum())))

        test_losses = np.zeros(len(test_loader))
        with torch.no_grad():
            for i, img in enumerate(test_loader):
                if isinstance(img, list):
                    img, augmented = img
                else:
                    augmented = img

                img, augmented = img.to(device), augmented.to(device)
                test_losses[i] = criterion(torch.sigmoid(model(img)), img if not invert else 1. - img)

        outputs.append([np.mean(train_losses), np.mean(test_losses)])

    return np.vstack(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gaussian_noise', type=float, default=0.5)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='projs')
    parser.add_argument('--no_skip', action='store_true')
    parser.add_argument('--invert', action='store_true')

    args = parser.parse_args()

    model = Unet(layers=[8, 16, 32, 64, 128], output_channels=1, skip=not args.no_skip)
    _, dataloader_train, _, dataloader_test = get_dataloaders_unsupervised(dpi=50,
                                                                           workers=2,
                                                                           image_folder=args.dataset,
                                                                           augmentations=AddGaussianNoise(
                                                                               std=args.gaussian_noise))

    train_test_losses = train_as_autoencoder(model, dataloader_train, dataloader_test, mode='train',
                                             num_epochs=args.epochs, device=args.device,
                                             lr=args.lr, invert=args.invert)

    if args.save is not None:
        save_dir, _ = os.path.split(args.save)
        (Path() / 'logs' / save_dir).mkdir(parents=True, exist_ok=True)
        (Path() / 'learned_models' / save_dir).mkdir(parents=True, exist_ok=True)

        np.savetxt(Path() / 'logs' / f'{args.save}.out', train_test_losses)
        torch.save(model.state_dict(), Path() / 'learned_models' / f'{args.save}.pt')
