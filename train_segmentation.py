import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import get_dataloaders_unsupervised, get_dataloaders_supervised
from losses import focal_loss, dice_loss
from unet import Unet, SkipType

import os

device = 'cuda'


def transfer_knowledge(model, knowledge_path, device=device):
    state_dict = torch.load(knowledge_path, map_location=device)
    del state_dict['final.weight']
    del state_dict['final.bias']
    model.load_state_dict(state_dict, strict=False)


def train_as_segmantation(model, data_loader, test_loader, mode='train', num_epochs=5, lr=1e-4, dice=None, focal=False,
                          device=device, checkpoint=None):
    if not (mode == 'train' or mode == 'test'):
        raise ValueError("mode should be 'train' or 'test'")

    model = model.to(device)
    if mode == 'train':
        model.train()
    else:
        model.eval()

    def criterion(x, mask):
        x = torch.sigmoid(x)
        result = F.binary_cross_entropy(x, mask.float())

        if focal:
            result = focal_loss(result, mask)

        if dice is not None and "weight" in dice:
            channels = dice['channels'] if 'channels' in dice else None
            result += dice["weight"] * dice_loss(x, mask, channels=channels)

        return result

    # criterion = dice_loss  # nn.BCEWithLogitsLoss()  # F.binary_cross_entropy_with_logits
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # torch.optim.SGD(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # length = len(dataloader)

    outputs = []
    # test_outputs = []
    for epoch in range(num_epochs):
        train_losses = np.zeros(len(data_loader))
        # losses_test = np.zeros(len(test_loader))

        for i, (img, mask) in enumerate(data_loader):
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()

            # img[img == 0] = 1e-7
            # img = 1 - img
            x = model(img)
            # x = torch.sigmoid(x)
            loss = criterion(x, mask)  # F.binary_cross_entropy(x, mask.float()) + criterion(x, mask)

            if mode == 'train':
                loss.backward()
                optimizer.step()

            train_losses[i] = loss.item()
            if i % 10 == 0 or i == len(data_loader) - 1:
                print('Epoch:{}/{}, Step:{}/{}, Loss:{:.6f}'.format(epoch + 1, num_epochs, i, len(data_loader),
                                                                    np.true_divide(train_losses.sum(),
                                                                                   (train_losses != 0).sum())))

        if checkpoint is not None:
            checkpoint(epoch, model)
            # if epoch % checkpoint == 0:
            #     torch.save(model.state_dict(), Path() / 'checkpoints' / )

        test_losses = np.zeros(len(test_loader))
        with torch.no_grad():
            for i, (img, mask) in enumerate(test_loader):
                img, mask = img.to(device), mask.to(device)
                test_losses[i] = criterion(model(img), mask)

        outputs.append([np.mean(train_losses), np.mean(test_losses)])
        # else:
        # losses_test = np.zeros(len(test_loader))

    return np.array(outputs)


def train_segmentation(args):
    skip_type = SkipType.SKIP if not args.no_skip else SkipType.NO_SKIP
    model = Unet(layers=[8, 16, 32, 64, 128], output_channels=11, skip=skip_type, dropout=args.dropout)

    if args.transfer is not None:
        transfer_knowledge(model, Path() / 'learned_models' / args.transfer, device=args.device)

    if args.load is not None:
        model.load_state_dict(torch.load(Path() / 'learned_models' / args.load, map_location=args.device))

    imgs, masks = 'projs', 'mask.zip'
    # if args.cutout:
    #     imgs, masks = 'projs_cutout', 'mask_cutout.zip'

    if args.projs is not None:
        imgs = args.projs
    if args.masks is not None:
        masks = args.masks

    dataset_train, dataloader_train, dataset_test, dataloader_test = get_dataloaders_supervised(root=args.root, image_folder=imgs, mask_folder=masks)

    save_dir, _ = os.path.split(args.save)
    if args.checkpoint != -1:
        (Path() / 'checkpoints' / save_dir).mkdir(parents=True, exist_ok=True)
    (Path() / 'logs' / save_dir).mkdir(parents=True, exist_ok=True)
    if not args.dont_save_model:
        (Path() / 'learned_models' / save_dir).mkdir(parents=True, exist_ok=True)

    def checkpoint(e, m):
        if args.checkpoint != -1 and e % args.checkpoint == 0:
            torch.save(m.state_dict(), Path() / 'checkpoints' / f'{args.save}_{e}epoch.pt')

    losses = train_as_segmantation(model, dataloader_train, dataloader_test, device=args.device, num_epochs=args.epochs, lr=args.lr, checkpoint=checkpoint)

    # if args.save is not None:
    np.savetxt(Path() / "logs" / f'{args.save}.out', losses)
    if not args.dont_save_model:
        torch.save(model.state_dict(), Path() / 'learned_models' / f'{args.save}.pt')


# if __name__ == '__main__':
#     test_on_cats_and_blueprints()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save', type=str, default="save")
    parser.add_argument('--load', type=str, default=None)
    # parser.add_argument('--resize', type=int, default=None)
    parser.add_argument('--transfer', type=str, default=None)
    parser.add_argument('--checkpoint', type=int, default=10)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--projs', type=str, default=None)
    parser.add_argument('--masks', type=str, default=None)
    parser.add_argument('--root', type=str, default=str(Path() / 'blueprints'))
    parser.add_argument('--dont_save_model', action='store_true')
    parser.add_argument('--no_skip', action='store_true')
    parser.add_argument('--dropout', action='store_true')

    args = parser.parse_args()

    train_segmentation(args)
