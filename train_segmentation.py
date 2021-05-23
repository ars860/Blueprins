import argparse
import time
from os.path import basename, splitext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import ConcatDataset, DataLoader

import augmentation
from dataset import get_dataloaders_unsupervised, get_dataloaders_supervised, BlueprintsSupervisedDataset
from iou import iou_multi_channel, iou_global
from losses import focal_loss, dice_loss
from unet import Unet, SkipType

import os

import wandb

device = 'cuda'


def transfer_knowledge(model, knowledge_path, device=device, freeze=False, random_decoder=False):
    state_dict = torch.load(knowledge_path, map_location=device)
    del state_dict['final.weight']
    del state_dict['final.bias']

    if random_decoder:
        for name in list(state_dict.keys()):
            if 'up' in name:
                del state_dict[name]

    model.load_state_dict(state_dict, strict=False)

    if freeze:
        for name, param in model.named_parameters():
            if 'down' in name:
                param.requires_grad = False


def transfer_knowledge_from_wandb(model, knowledge_path, run, device=device, freeze=False, random_decoder=False):
    assert not random_decoder

    artifact = run.use_artifact(knowledge_path, type='model')
    artifact_dir = artifact.download()
    artifact_file = next(iter((Path() / artifact_dir).glob('*')))

    state_dict = torch.load(artifact_file, map_location=device)

    del state_dict['final.weight']
    del state_dict['final.bias']
    model.load_state_dict(state_dict, strict=False)

    if freeze:
        for name, param in model.named_parameters():
            if 'down' in name:
                param.requires_grad = False


def train_as_segmantation(model, data_loader, test_loader, mode='train', num_epochs=5, lr=1e-4, bce=True, dice=None, focal=False,
                          device=device, checkpoint=None, no_wandb=False, optim='adam', sched=None, iou_c=False, watch=False):
    if not (mode == 'train' or mode == 'test'):
        raise ValueError("mode should be 'train' or 'test'")

    model = model.to(device)
    if mode == 'train':
        model.train()
    else:
        model.eval()

    if not no_wandb and watch:
        wandb.watch(model, log_freq=100)

    def criterion(x, mask, channels=None):
        # x = torch.sigmoid(x)

        if channels is not None:
            x = x[:, channels]

        result = None
        if bce:
            result = F.binary_cross_entropy(x, mask.float())

        if focal:
            if result is None:
                result = F.binary_cross_entropy(x, mask.float())

            result = focal_loss(result, mask)

        if dice is not None and "weight" in dice:
            dice_channels = dice['channels'] if 'channels' in dice else None
            if result is None:
                result = 0
            result += dice["weight"] * dice_loss(x, mask, channels=dice_channels)

        return result

    # criterion = dice_loss  # nn.BCEWithLogitsLoss()  # F.binary_cross_entropy_with_logits
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # torch.optim.SGD(model.parameters(), lr=lr)
    if optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    scheduler = None
    if sched is not None:
        if sched == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        if sched == 'step_lr':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # length = len(dataloader)

    outputs = []
    # test_outputs = []
    for epoch in range(num_epochs):
        train_losses = np.zeros(len(data_loader))
        model.train()
        # losses_test = np.zeros(len(test_loader))

        for i, smth in enumerate(data_loader):
            channels = None
            if len(smth) == 2:
                img, mask = smth
            else:
                channels, img, mask = smth
                channels = list(map(lambda ch: ch.item(), channels))

                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / args.reduce_lr_on_additional

            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()

            # img[img == 0] = 1e-7
            # img = 1 - img
            x = model(img)
            # x = torch.sigmoid(x)
            loss = criterion(x, mask, channels=channels)  # F.binary_cross_entropy(x, mask.float()) + criterion(x, mask)

            if mode == 'train':
                loss.backward()
                optimizer.step()

            train_losses[i] = loss.item()
            if i % 10 == 0 or i == len(data_loader) - 1:
                print('Epoch:{}/{}, Step:{}/{}, Loss:{:.6f}'.format(epoch + 1, num_epochs, i, len(data_loader),
                                                                    np.true_divide(train_losses.sum(),
                                                                                   (train_losses != 0).sum())))

            if channels is not None:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] * args.reduce_lr_on_additional

        if scheduler is not None:
            scheduler.step(np.true_divide(train_losses.sum(), (train_losses != 0).sum()))

        if checkpoint is not None:
            checkpoint(epoch, model)
            # if epoch % checkpoint == 0:
            #     torch.save(model.state_dict(), Path() / 'checkpoints' / )

        test_losses = np.zeros(len(test_loader))
        # test_ious = np.zeros(len(test_loader))
        with torch.no_grad():
            model.eval()
            for i, smth in enumerate(test_loader):
                channels = None
                if len(smth) == 2:
                    img, mask = smth
                else:
                    channels, img, mask = smth
                    channels = list(map(lambda ch: ch.item(), channels))

                img, mask = img.to(device), mask.to(device)
                processed = model(img)
                test_losses[i] = criterion(processed, mask, channels=channels)
                # test_ious[i] = np.mean(iou_multi_channel(processed, mask))

        train_losses, test_losses, test_ious = np.mean(train_losses), np.mean(test_losses), iou_global(test_loader, model=model, device=device, concat=iou_c)  # np.mean(test_ious)

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
        config.scheduler = args.scheduler
        config.iou_concat = args.iou_concat
        config.transfer_freeze = args.transfer_freeze
        config.hide_aug = args.hide_aug
        config.random_decoder = args.random_decoder
        config.additional_dataset = args.additional_roots is not None and len(args.additional_roots) != 0
        config.reduce_lr_on_additional = args.reduce_lr_on_additional

        if args.run_name is not None:
            wandb.run.name = args.run_name
            wandb.run.save()

    skip_type = SkipType.SKIP if not args.no_skip else SkipType.NO_SKIP

    model = Unet(layers=args.layers, output_channels=11, skip=skip_type, dropout=args.dropout)

    if args.transfer is not None:
        if args.load_from_wandb:
            transfer_knowledge_from_wandb(model, f'ars860/diplom_autoencoders/{args.transfer}', run=run,
                                          device=args.device, freeze=args.transfer_freeze, random_decoder=args.random_decoder)
        else:
            transfer_knowledge(model, Path() / 'learned_models' / args.transfer, device=args.device, freeze=args.transfer_freeze, random_decoder=args.random_decoder)

    if args.load is not None:
        model.load_state_dict(torch.load(Path() / args.load, map_location=args.device))

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
    if args.hide_aug is not None and args.hide_aug != 0:
        transforms.append(augmentation.HideRectangle(min_cnt=1, max_cnt=args.hide_aug, channels=[2, 3, 4, 5], p=0.5))
    if args.cutout_cnt != 0 and args.cutout_p != 0:
        transforms.append(A.Cutout(num_holes=args.cutout_cnt, p=args.cutout_p))
    transforms.append(ToTensorV2())
    transforms = A.Compose(transforms)

    # transforms = A.Compose([A.VerticalFlip(), A.HorizontalFlip(), A.SmallestMaxSize(256), ToTensorV2()])
    # if args.cutout_cnt != 0:
    #     transforms = A.Compose([A.VerticalFlip(), A.HorizontalFlip(), A.SmallestMaxSize(256), A.Cutout(num_holes=args.cutout_cnt, p=args.cutout_p), ToTensorV2()])

    # dataset_train, dataloader_train, dataset_test, dataloader_test = get_dataloaders_supervised(root=args.root,
    #                                                                                             image_folder=imgs,
    #                                                                                             mask_folder=masks,
    #                                                                                             transforms=transforms)

    main_dataset_train = BlueprintsSupervisedDataset(mode='train', fraction=0.9, root=args.root, image_folder=imgs, mask_folder=masks, transforms=transforms)
    main_dataset_test = BlueprintsSupervisedDataset(mode='test', fraction=0.9, root=args.root, image_folder=imgs, mask_folder=masks, transforms=transforms)
    datasets_train, datasets_test = [main_dataset_train], [main_dataset_test]
    if args.additional_roots != ['']:
        for root in args.additional_roots:
            dataset_train = BlueprintsSupervisedDataset(root, imgs, masks, mode='train', transforms=transforms, channels=args.additional_roots_channels)
            # TODO: think
            # dataset_test = BlueprintsSupervisedDataset(root, imgs, masks, mode='test', transforms=transforms, channels=args.additional_roots_channels)
            datasets_train.append(dataset_train)
            # datasets_test.append(dataset_test)

    dataset_train = ConcatDataset(datasets_train)
    dataloader_train = DataLoader(dataset_train, num_workers=2)
    dataset_test = ConcatDataset(datasets_test)
    dataloader_test = DataLoader(dataset_test, num_workers=2)

    if args.save is not None:
        save_dir, _ = os.path.split(args.save)
        if args.checkpoint != -1:
            (Path() / 'checkpoints' / save_dir).mkdir(parents=True, exist_ok=True)
        (Path() / 'logs' / save_dir).mkdir(parents=True, exist_ok=True)
        if not args.dont_save_model:
            (Path() / 'learned_models' / save_dir).mkdir(parents=True, exist_ok=True)

    def checkpoint(e, m):
        if args.save is not None:
            if args.checkpoint != -1 and (e + 1) % args.checkpoint == 0:
                torch.save(m.state_dict(), Path() / 'checkpoints' / f'{args.save}_{e}epoch.pt')

                if not args.no_wandb:
                    artifact = wandb.Artifact(f'checkpoint_{e}', type='checkpoint')
                    artifact.add_file(str(Path() / 'checkpoints' / f'{args.save}_{e}epoch.pt'))
                    run.log_artifact(artifact)

    dice = list(filter(lambda s: 'dice' in s, args.loss.split('_')))
    if len(dice) == 1:
        dice, = dice
        dice_channels = list(map(int, filter(lambda s: s != 'dice', dice.split('|'))))
        dice = {"weight": 1.0, "channels": None if dice_channels == [] else dice_channels}
    else:
        dice = None

    losses = train_as_segmantation(model, dataloader_train, dataloader_test, device=args.device, num_epochs=args.epochs,
                                   lr=args.lr, checkpoint=checkpoint, no_wandb=args.no_wandb, optim=args.optimizer,
                                   bce='bce' in args.loss, focal='focal' in args.loss, dice=dice,
                                   sched=args.scheduler, iou_c=args.iou_concat, watch=not args.transfer_freeze)

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
    parser.add_argument('--cutout_cnt', type=int, default=None)
    parser.add_argument('--cutout_p', type=float, default=None)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--vh', action='store_true')
    parser.add_argument('--load_from_wandb', action='store_true')
    parser.add_argument('--optimizer', type=str, default='adam')

    parser.add_argument('--layers', type=int, nargs='+', default=[8, 16, 32, 64, 128])

    parser.add_argument('--vh_', type=lambda s: s == 'true', default=None)
    parser.add_argument('--skip_type', type=str, default=None)
    parser.add_argument('--loss', type=str, default='bce')

    parser.add_argument('--scheduler', type=str, default="no")
    parser.add_argument('--iou_concat', type=lambda s: s == 'true', default=True)
    parser.add_argument('--transfer_freeze', type=lambda s: s == 'true', default=None)
    parser.add_argument('--random_decoder', type=lambda s: s == 'true', default=None)

    parser.add_argument('--config', type=str, default=None)

    parser.add_argument('--hide_aug', type=int, default=0)

    parser.add_argument('--cutout_config', type=str, default=None)

    parser.add_argument('--additional_roots', type=str, nargs='+', default=[str(Path() / 'blueprints' / 'new')])
    parser.add_argument('--additional_roots_channels', type=lambda s: list(map(int, s.split('_'))), default=[2, 3, 4, 5, 6, 9])

    parser.add_argument('--reduce_lr_on_additional', type=int, default=1)

    args = parser.parse_args()

    if args.config is not None:
        assert args.transfer is None
        assert args.transfer_freeze is None
        assert args.skip_type is None
        assert args.random_decoder is None

        if args.config == 'skip':
            args.transfer = None
            args.skip_type = 'skip'
        elif args.config == 'no_skip':
            args.transfer = None
            args.skip_type = 'no_skip'
        else:
            if 'no_skip' in args.config:
                args.skip_type = 'no_skip'

            splitted = args.config.split('__')
            args.transfer = next(filter(lambda s: s != "freeze" and s != "random_decoder", splitted))

            if 'freeze' in splitted:
                args.transfer_freeze = True
            if 'random_decoder' in splitted:
                args.random_decoder = True

    if args.cutout_config is not None:
        assert args.cutout_cnt is None
        assert args.cutout_p is None

        p, cnt = args.cutout_config.split('_')
        p, cnt = float(p), int(cnt)

        args.cutout_cnt, args.cutout_p = cnt, p

    if args.cutout_cnt is None:
        args.cutout_cnt = 0

    if args.cutout_p is None:
        args.cutout_p = 0

    if args.vh_ is not None:
        args.vh = args.vh_

    if args.skip_type is not None:
        if args.skip_type == 'no_skip':
            args.no_skip = True

    if args.transfer == '':
        args.transfer = None

    if args.scheduler == '':
        args.transfer = None

    if args.transfer is not None:
        if 'no_skip' in args.transfer:
            args.no_skip = True
        else:
            args.no_skip = False

    if args.transfer_freeze is None:
        args.transfer_freeze = False

    # if args.save is None:
    #     args.save = f'sweep/{args.lr}_{args.epochs}epochs' \
    #                 f'{f"_transfer_{splitext(basename(args.transfer))[0]}" if args.transfer is not None else ""}' \
    #                 f'{"_no_skip" if args.no_skip and args.transfer is None else ""}'

    # print(args.save)

    train_segmentation(args)
