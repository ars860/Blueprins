import argparse
import time
from os.path import basename, splitext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import get_dataloaders_unsupervised, get_dataloaders_supervised
from iou import iou_multi_channel
from losses import focal_loss, dice_loss
from unet import Unet, SkipType

import os

import wandb

device = 'cuda'


def transfer_knowledge(model, knowledge_path, device=device):
    state_dict = torch.load(knowledge_path, map_location=device)
    del state_dict['final.weight']
    del state_dict['final.bias']
    model.load_state_dict(state_dict, strict=False)


def transfer_knowledge_from_wandb(model, knowledge_path, run, device=device):
    artifact = run.use_artifact(knowledge_path, type='model')
    artifact_dir = artifact.download()
    artifact_file = next(iter((Path() / artifact_dir).glob('*')))

    state_dict = torch.load(artifact_file, map_location=device)

    del state_dict['final.weight']
    del state_dict['final.bias']
    model.load_state_dict(state_dict, strict=False)


def train_as_segmantation(model, data_loader, test_loader, mode='train', num_epochs=5, lr=1e-4, bce=True, dice=None, focal=False,
                          device=device, checkpoint=None, no_wandb=False, optim='adam'):
    if not (mode == 'train' or mode == 'test'):
        raise ValueError("mode should be 'train' or 'test'")

    model = model.to(device)
    if mode == 'train':
        model.train()
    else:
        model.eval()

    if not no_wandb:
        wandb.watch(model, log_freq=100)

    def criterion(x, mask):
        # x = torch.sigmoid(x)
        result = None
        if bce:
            result = F.binary_cross_entropy(x, mask.float())

        if focal:
            if result is None:
                result = F.binary_cross_entropy(x, mask.float())

            result = focal_loss(result, mask)

        if dice is not None and "weight" in dice:
            channels = dice['channels'] if 'channels' in dice else None
            if result is None:
                result = 0
            result += dice["weight"] * dice_loss(x, mask, channels=channels)

        return result

    # criterion = dice_loss  # nn.BCEWithLogitsLoss()  # F.binary_cross_entropy_with_logits
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # torch.optim.SGD(model.parameters(), lr=lr)
    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
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
        test_ious = np.zeros(len(test_loader))
        with torch.no_grad():
            for i, (img, mask) in enumerate(test_loader):
                img, mask = img.to(device), mask.to(device)
                processed = model(img)
                test_losses[i] = criterion(processed, mask)
                test_ious[i] = np.mean(iou_multi_channel(processed, mask))

        train_losses, test_losses, test_ious = np.mean(train_losses), np.mean(test_losses), np.mean(test_ious)

        print(train_losses, test_losses, test_ious)

        outputs.append([train_losses, test_losses, test_ious])
        if not no_wandb:
            wandb.log({"train_loss": train_losses,
                       "test_loss": test_losses,
                       "test_iou": test_ious})
        # else:
        # losses_test = np.zeros(len(test_loader))

    return np.array(outputs)


# previously vh = True
def train_segmentation(args):
    # if args.run_name is None:
    #     args.run_name = f'{args.lr}_{args.epochs}epochs{f"_transfer_{args.transfer}" if args.transfer is not None else ""}{"_no_skip" if args.no_skip else ""}'

    run = None
    if not args.no_wandb:
        run = wandb.init(project='diplom_segmentation', entity='ars860')
        config = wandb.config
        config.lr = args.lr
        config.epochs = args.epochs
        config.no_skip = args.no_skip
        config.transfer = args.transfer
        config.load = args.load
        config.dropout = args.dropout
        config.cutout_cnt = args.cutout_cnt
        config.cutout_p = args.cutout_p
        config.vh = args.vh
        config.optimizer = args.optimizer
        config.loss = args.loss

        if args.run_name is not None:
            wandb.run.name = args.run_name
            wandb.run.save()

    skip_type = SkipType.SKIP if not args.no_skip else SkipType.NO_SKIP

    model = Unet(layers=args.layers, output_channels=11, skip=skip_type, dropout=args.dropout)

    if args.transfer is not None:
        if args.load_from_wandb:
            transfer_knowledge_from_wandb(model, f'ars860/diplom_autoencoders/{args.transfer}', run=run,
                                          device=args.device)
        else:
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

    transforms = []
    if args.vh:
        transforms = [A.VerticalFlip(), A.HorizontalFlip()]
    transforms.append(A.SmallestMaxSize(256))
    if args.cutout_cnt != 0 and args.cutout_p != 0:
        transforms.append(A.Cutout(num_holes=args.cutout_cnt, p=args.cutout_p))
    transforms.append(ToTensorV2())
    transforms = A.Compose(transforms)

    # transforms = A.Compose([A.VerticalFlip(), A.HorizontalFlip(), A.SmallestMaxSize(256), ToTensorV2()])
    # if args.cutout_cnt != 0:
    #     transforms = A.Compose([A.VerticalFlip(), A.HorizontalFlip(), A.SmallestMaxSize(256), A.Cutout(num_holes=args.cutout_cnt, p=args.cutout_p), ToTensorV2()])

    dataset_train, dataloader_train, dataset_test, dataloader_test = get_dataloaders_supervised(root=args.root,
                                                                                                image_folder=imgs,
                                                                                                mask_folder=masks,
                                                                                                transforms=transforms)

    if args.save is not None:
        save_dir, _ = os.path.split(args.save)
        if args.checkpoint != -1:
            (Path() / 'checkpoints' / save_dir).mkdir(parents=True, exist_ok=True)
        (Path() / 'logs' / save_dir).mkdir(parents=True, exist_ok=True)
        if not args.dont_save_model:
            (Path() / 'learned_models' / save_dir).mkdir(parents=True, exist_ok=True)

    def checkpoint(e, m):
        if args.save is not None:
            if args.checkpoint != -1 and e % args.checkpoint == 0:
                torch.save(m.state_dict(), Path() / 'checkpoints' / f'{args.save}_{e}epoch.pt')

                if not args.no_wandb:
                    artifact = wandb.Artifact(f'checkpoint_{e}', type='checkpoint')
                    artifact.add_file(str(Path() / 'checkpoints' / f'{args.save}_{e}epoch.pt'))
                    run.log_artifact(artifact)

    losses = train_as_segmantation(model, dataloader_train, dataloader_test, device=args.device, num_epochs=args.epochs,
                                   lr=args.lr, checkpoint=checkpoint, no_wandb=args.no_wandb, optim=args.optimizer,
                                   bce='bce' in args.loss, dice={"weight": 1.0} if 'dice' in args.loss else None)

    if args.save is not None:
        np.savetxt(Path() / "logs" / f'{args.save}.out', losses)
        if not args.dont_save_model:
            torch.save(model.state_dict(), Path() / 'learned_models' / f'{args.save}.pt')

    if not args.no_wandb:
        artifact = wandb.Artifact('model', type='model')
        with artifact.new_file('sweep_model.pt', mode='wb') as f:
            torch.save(model.state_dict(), f)
        # artifact.add_file(str(Path() / 'learned_models' / f'{args.save}.pt'))
        run.log_artifact(artifact)


# if __name__ == '__main__':
#     test_on_cats_and_blueprints()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--load', type=str, default=None)
    # parser.add_argument('--resize', type=int, default=None)
    parser.add_argument('--transfer', type=str, default=None)
    parser.add_argument('--checkpoint', type=int, default=-1)
    # parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--projs', type=str, default=None)
    parser.add_argument('--masks', type=str, default=None)
    parser.add_argument('--root', type=str, default=str(Path() / 'blueprints'))
    parser.add_argument('--dont_save_model', action='store_true')
    parser.add_argument('--no_skip', action='store_true')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--cutout_cnt', type=int, default=0)
    parser.add_argument('--cutout_p', type=float, default=0)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--vh', action='store_true')
    parser.add_argument('--load_from_wandb', action='store_true')
    parser.add_argument('--optimizer', type=str, default='adam')

    parser.add_argument('--layers', type=int, nargs='+', default=[8, 16, 32, 64, 128])

    parser.add_argument('--vh_', type=lambda s: s == 'true', default=None)
    parser.add_argument('--skip_type', type=str, default='skip')
    parser.add_argument('--loss', type=str, default='bce')

    args = parser.parse_args()

    if args.vh_ is not None:
        args.vh = args.vh_

    if args.skip_type is not None:
        if args.skip_type == 'no_skip':
            args.no_skip = True

    if args.transfer == '':
        args.transfer = None

    if args.transfer is not None:
        if 'no_skip' in args.transfer:
            args.no_skip = True
        else:
            args.no_skip = False

    # if args.save is None:
    #     args.save = f'sweep/{args.lr}_{args.epochs}epochs' \
    #                 f'{f"_transfer_{splitext(basename(args.transfer))[0]}" if args.transfer is not None else ""}' \
    #                 f'{"_no_skip" if args.no_skip and args.transfer is None else ""}'

    # print(args.save)
    train_segmentation(args)
