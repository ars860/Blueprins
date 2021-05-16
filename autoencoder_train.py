import argparse
import os
from pathlib import Path

import numpy as np
import torch.nn as nn
import torch

from augmentation import AddGaussianNoise
from dataset import get_dataloaders_unsupervised
from train_segmentation import device
from unet import Unet, SkipType

import wandb


def train_as_autoencoder(model, data_loader, test_loader, num_epochs=5, mode=None, device=device, lr=1e-3,
                         invert=False, plot_each=500, no_sigmoid=False):
    if not (mode == 'train' or mode == 'test'):
        raise ValueError("mode should be 'train' or 'test'")

    model = model.to(device)
    if mode == 'train':
        model.train()
    else:
        model.eval()

    # wandb.watch(model, log_freq=100)

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

            x = model(augmented) if not no_sigmoid else model.forward_vars(augmented)['res']
            # x = torch.sigmoid(x)
            loss = criterion(x, img if not invert else 1. - img)

            if mode == 'train':
                loss.backward()
                optimizer.step()

            train_losses[i] = loss.item()
            # if (i + 1) % 10 == 0 or i == len(data_loader) - 1:
            #     print('Epoch:{}/{}, Step:{}/{}, Loss:{:.4f}'.format(epoch + 1, num_epochs, i + 1, len(data_loader),
            #                                                         train_losses / (i + 1)))

            if plot_each is not None and (i + 1) % plot_each == 0 or i == len(data_loader) - 1:
                test_losses = np.zeros(len(test_loader))
                with torch.no_grad():
                    for j, img in enumerate(test_loader):
                        if isinstance(img, list):
                            img, augmented = img
                        else:
                            augmented = img

                        img, augmented = img.to(device), augmented.to(device)
                        x = model(img) if not no_sigmoid else model.forward_vars(augmented)['res']
                        test_losses[j] = criterion(x, img if not invert else 1. - img)

                wandb.log({"train_loss": np.true_divide(train_losses.sum(), (train_losses != 0).sum()),
                           "test_loss": np.mean(test_losses)})
                outputs.append([np.true_divide(train_losses.sum(), (train_losses != 0).sum()), np.mean(test_losses)])

                print('Epoch:{}/{}, Step:{}/{}, TrainLoss:{:.4f}, TestLoss:{}'.format(epoch + 1, num_epochs, i + 1,
                                                                                      len(data_loader),
                                                                                      np.true_divide(
                                                                                          train_losses.sum(),
                                                                                          (train_losses != 0).sum()),
                                                                                      np.mean(test_losses)))

    return np.vstack(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gaussian_noise', type=float, default=None)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='projs')
    parser.add_argument('--no_skip', action='store_true')
    parser.add_argument('--zero_skip', action='store_true')
    parser.add_argument('--invert', action='store_true')
    parser.add_argument('--shuffle_seed', type=int, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true')

    parser.add_argument('--layers', type=int, nargs='+', default=[8, 16, 32, 64, 128])

    parser.add_argument('--no_sigmoid', type=lambda s: s == 'true', default=None)

    args = parser.parse_args()

    if args.no_skip and args.zero_skip:
        raise ValueError('Only one skip type can be specified: <no_skip> or <zero_skip>')

    skip_type = SkipType.SKIP
    if args.no_skip:
        skip_type = SkipType.NO_SKIP
    if args.zero_skip:
        skip_type = SkipType.ZERO_SKIP

    run = None
    if not args.no_wandb:
        run = wandb.init(project='diplom_autoencoders', entity='ars860')
        config = wandb.config
        config.learning_rate = args.lr
        config.epochs = args.epochs
        config.invert = args.invert
        config.no_skip = args.no_skip
        config.zero_skip = args.zero_skip
        config.layers = args.layers
        config.gaussian_noise = args.gaussian_noise
        config.no_sigmoid = args.no_sigmoid

        if args.run_name is not None:
            wandb.run.name = args.run_name
            wandb.run.save()

    model = Unet(layers=args.layers, output_channels=1, skip=skip_type)

    if not args.no_wandb:
        wandb.watch(model, log_freq=100)

    _, dataloader_train, _, dataloader_test = get_dataloaders_unsupervised(dpi=50,
                                                                           workers=2,
                                                                           image_folder=args.dataset,
                                                                           shuffle_seed=args.shuffle_seed,
                                                                           augmentations=None if args.gaussian_noise is None
                                                                           else AddGaussianNoise(
                                                                               std=args.gaussian_noise))

    train_test_losses = train_as_autoencoder(model, dataloader_train, dataloader_test, mode='train',
                                             num_epochs=args.epochs, device=args.device,
                                             lr=args.lr, invert=args.invert, no_sigmoid=args.no_sigmoid)

    if args.save is not None:
        save_dir, _ = os.path.split(args.save)
        (Path() / 'logs' / save_dir).mkdir(parents=True, exist_ok=True)
        (Path() / 'learned_models' / save_dir).mkdir(parents=True, exist_ok=True)

        np.savetxt(Path() / 'logs' / f'{args.save}.out', train_test_losses)
        torch.save(model.state_dict(), Path() / 'learned_models' / f'{args.save}.pt')

    if not args.no_wandb:
        artifact = wandb.Artifact('model', type='model')
        with artifact.new_file('sweep_model.pt', mode='wb') as f:
            torch.save(model.state_dict(), f)
        # artifact.add_file(str(Path() / 'learned_models' / f'{args.save}.pt'))
        run.log_artifact(artifact)
