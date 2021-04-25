from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from dataset import get_dataloaders_unsupervised, get_dataloaders_supervised
from unet import Unet

device = 'cpu'


def dice_loss(pred, target, smooth=1., ignored_channels=None):
    if ignored_channels is None:
        ignored_channels = []

    channels = [channel for channel in range(pred.shape[1]) if channel not in ignored_channels]  # list(set(range()) - set(ignored_channels))

    pred = pred[:, channels].contiguous()
    target = target[:, channels].contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def transfer_knowledge(model, knowledge_path, device=device):
    state_dict = torch.load(knowledge_path, map_location=device)
    del state_dict['final.weight']
    del state_dict['final.bias']
    model.load_state_dict(state_dict, strict=False)


def train_as_segmantation(model, data_loader, mode='train', num_epochs=5, lr=1e-4):
    if not (mode == 'train' or mode == 'test'):
        raise ValueError("mode should be 'train' or 'test'")

    model = model.to(device)
    if mode == 'train':
        model.train()
    else:
        model.eval()

    criterion = dice_loss  # nn.BCEWithLogitsLoss()  # F.binary_cross_entropy_with_logits
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # outputs = []
    for epoch in range(num_epochs):
        for i, (img, mask) in enumerate(data_loader):
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()

            # img[img == 0] = 1e-7
            # img = 1 - img
            x = model(img)
            x = torch.sigmoid(x)
            loss = F.binary_cross_entropy(x, mask.float()) + dice_loss(x, mask)

            if mode == 'train':
                loss.backward()
                optimizer.step()

            if i % 10 == 0 or i == len(data_loader) - 1:
                print('Epoch:{}/{}, Step:{}/{}, Loss:{:.4f}'.format(epoch + 1, num_epochs, i, len(data_loader),
                                                                    float(loss)))
        if mode == 'train':
            scheduler.step()

        # outputs.append((epoch, img, mask, x))

    # return outputs


def test_on_cats_and_blueprints():
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    from pathlib import Path
    import torch.nn as nn

    model = Unet(layers=[16, 32, 64, 128], output_channels=3)
    transfer_knowledge(model, Path() / 'learned_models' / 'unet_16_autoencoder.pt')
    dataset_train, dataloader_train, dataset_test, dataloader_test = get_dataloaders_unsupervised(dpi=25)

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
    ax2.imshow(img_decoded.cpu().detach().numpy().squeeze()[0, :])

    img_decoded = img_decoded.cpu().detach().numpy().squeeze()
    img_decoded = img_decoded.transpose(1, 2, 0)  # np.swapaxes(img_decoded, 0, 2)
    img_decoded = np.uint8((img_decoded + np.min(img_decoded)) / (np.max(img_decoded) + np.min(img_decoded)) * 255)
    Image.fromarray(img_decoded, 'RGB').show()

    # fig.colorbar(a, ax=fig)
    fig.suptitle(f'Shape: {img.shape} Loss: {loss}', fontsize=30)
    plt.show()
    # train_as_autoencoder(model, dataloader_train)


def train_segmentation():
    model = Unet(layers=[8, 16, 32, 64, 128], output_channels=11)
    # transfer_knowledge(model, Path() / 'learned_models' / 'unet_16_autoencoder.pt')

    dataset_train, dataloader_train, dataset_test, dataloader_test = get_dataloaders_supervised()

    train_as_segmantation(model, dataloader_train)


if __name__ == '__main__':
    train_segmentation()
