import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import get_dataloaders_unsupervised, get_dataloaders_supervised
from losses import focal_loss, dice_loss
from unet import Unet

device = 'cuda'


def transfer_knowledge(model, knowledge_path, device=device):
    state_dict = torch.load(knowledge_path, map_location=device)
    del state_dict['final.weight']
    del state_dict['final.bias']
    model.load_state_dict(state_dict, strict=False)


def train_as_segmantation(model, data_loader, test_loader, mode='train', num_epochs=5, lr=1e-4, dice=None, focal=False,
                          device=device):
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

        test_losses = np.zeros(len(test_loader))
        with torch.no_grad():
            for i, img in enumerate(test_loader):
                img = img.to(device)
                test_losses[i] = criterion(model(img), img)

        outputs.append([np.mean(train_losses), np.mean(test_losses)])
        # else:
        # losses_test = np.zeros(len(test_loader))

    return np.array(outputs)


def test_on_cats_and_blueprints():
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    from pathlib import Path
    import torch.nn as nn

    model = Unet(layers=[8, 16, 32, 64, 128], output_channels=1)
    transfer_knowledge(model, Path() / 'learned_models' / 'autoencoder_noise_0.1_sgd.pt')
    dataset_train, dataloader_train, dataset_test, dataloader_test = get_dataloaders_unsupervised(dpi=50)

    criterion = nn.MSELoss()

    model.eval()
    # img = next(iter(dataloader_train))
    img = torch.Tensor(np.array(Image.open('E:/acady/Desktop/83211.jpg').convert('L')).reshape([1, 1, 173, -1]))
    img_decoded = model(img)

    print(img.shape)
    print(img_decoded.shape)
    loss = criterion(img, img_decoded)
    print(loss)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[30, 20])
    a = ax1.imshow(img.squeeze())
    ax2.imshow(img_decoded.cpu().detach().numpy().squeeze())

    # img_decoded = img_decoded.cpu().detach().numpy().squeeze()
    # img_decoded = img_decoded.transpose(1, 2, 0)  # np.swapaxes(img_decoded, 0, 2)
    # img_decoded = np.uint8((img_decoded + np.min(img_decoded)) / (np.max(img_decoded) + np.min(img_decoded)) * 255)
    # Image.fromarray(img_decoded, 'RGB').show()

    # fig.colorbar(a, ax=fig)
    fig.suptitle(f'Shape: {img.shape} Loss: {loss}', fontsize=30)
    plt.show()
    # train_as_autoencoder(model, dataloader_train)


def train_segmentation(args):
    model = Unet(layers=[8, 16, 32, 64, 128], output_channels=11)

    if args.transfer is not None:
        transfer_knowledge(model, Path() / 'learned_models' / args.transfer, device=args.device)

    if args.load is not None:
        model.load_state_dict(torch.load(Path() / 'learned_models' / args.load, map_location=args.device))

    dataset_train, dataloader_train, dataset_test, dataloader_test = get_dataloaders_supervised()

    losses = train_as_segmantation(model, dataloader_train, dataloader_test, device=args.device, num_epochs=args.epochs, lr=args.lr)

    # if args.save is not None:
    np.savetxt(Path() / "logs" / f'{args.save}.out', losses)
    torch.save(model.state_dict(), Path() / 'learned_models' / f'{args.save}.pt')


# if __name__ == '__main__':
#     test_on_cats_and_blueprints()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save', type=str, default="save")
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--resize', type=int, default=None)
    parser.add_argument('--transfer', type=str, default=None)

    args = parser.parse_args()

    train_segmentation(args)
