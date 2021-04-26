import argparse
import itertools
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn

from dataset import get_dataloaders_supervised
from train import device
from unet import Unet


def plot_sample(model, index, dataloader, samples=None, mode='show', filter=None, name_prefix="sample_",
                path=Path() / 'pics', device=device):
    if not (mode == 'show' or mode == 'save'):
        raise ValueError('mode should be "save" or "show"')

    model.eval()

    if samples is None:
        samples = list(itertools.islice((iter(dataloader)), index + 1))

    input_img, mask = samples[index]

    result = nn.Sigmoid()(model(input_img.to(device)).squeeze()).cpu().detach().numpy()

    if filter is not None:
        result[result >= filter] = 1
        result[result < filter] = 0

    fig, ax = plt.subplots(12, 2, figsize=[30, 100])
    ax[0][0].imshow(input_img.squeeze())
    # ax2.imshow(mask.squeeze())
    mask = mask.squeeze()
    for i in range(11):
        ax[1 + i][0].imshow(result[i, :])
        ax[1 + i][1].imshow(mask[i, :])

    if mode == 'show':
        plt.show()
    else:
        save_path = path / name_prefix / f'{index}.png'
        plt.savefig(save_path)
        print(f'saved: {save_path}')


def plot_first_ten(model, dataloader, mode, name_prefix="sample_", device='cpu'):
    samples = list(itertools.islice((iter(dataloader)), 10))

    for i in range(10):
        plot_sample(model, i, dataloader, mode=mode, name_prefix=name_prefix, samples=samples, device=device)


# matplotlib.use('module://backend_interagg')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda:2')
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--mode", type=str, default='show')
    parser.add_argument("--model", type=str, default='learned_models/without_transfer.pt')
    parser.add_argument("--save_name", type=str, default='pics/saved')

    args = parser.parse_args()

    model = Unet(layers=[8, 16, 32, 64, 128], output_channels=11).to(args.device)
    model.load_state_dict(torch.load(args.model, map_location=args.device))

    _, dataloader_train, _, dataloader_test = get_dataloaders_supervised()

    plot_first_ten(model, dataloader_test if args.test else dataloader_train, args.mode, args.save_name, device=args.device)
