import torch
import numpy as np

from dataset import get_dataloaders_supervised
from unet import Unet

SMOOTH = 1e-6


def iou_no_batch(outputs: torch.Tensor, labels: torch.Tensor):
    intersection = (outputs & labels).float().sum((0, 1))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((0, 1))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    # thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10  # This is equal to comparing with thresolds

    # assert thresholded.shape == ()
    return iou.item()


def iou_concat(outputs, labels, threshold=0.5):
    outputs, labels = outputs.squeeze(), labels.squeeze().bool()

    outputs, labels = torch.cat([output for output in outputs], dim=1) > threshold, torch.cat([label for label in labels], dim=1)

    return iou_no_batch(outputs, labels)


def iou_multi_channel(outputs, labels, threshold=0.5):
    outputs, labels = outputs.squeeze(), labels.squeeze().bool()
    if len(outputs.shape) != 3:
        raise ValueError('Only no batch_size supported')

    # outputs = outputs.squeeze(0)
    iou_channels = []
    for i in range(outputs.shape[0]):
        iou_channels.append(iou_no_batch(outputs[i, :] > threshold, labels[i, :]))

    return np.array(iou_channels)


def iou_global(dataloader, model, device, concat=True):
    with torch.no_grad():
        model = model.to(device)
        model.eval()

        if not concat:
            ious = [[] for _ in range(next(iter(dataloader))[1].shape[1])]
            for smth in dataloader:
                channels = None
                if len(smth) == 2:
                    img, mask = smth
                else:
                    channels, img, mask = smth
                    channels = list(map(lambda ch: ch.item(), channels))

                img, mask = img.to(device), mask.to(device)

                result = model(img)

                iou = iou_multi_channel(result, mask)

                for i, item in enumerate(iou):
                    ious[i].append(item)

            ious = np.vstack(ious)
            return np.mean(ious, axis=1)

        ious = []
        for smth in dataloader:
            # channels = None
            # if len(smth) == 2:
            img, mask = smth
            # else:
            #     channels, img, mask = smth
            #     channels = list(map(lambda ch: ch.item(), channels))

            img, mask = img.to(device), mask.to(device)

            result = model(img)

            # if channels is not None:
            #     channel2mask = dict(list(zip(map(lambda ch: ch.item(), channels), [mask[0, i] for i in range(mask.shape[1])])))
            #     mask = torch.unsqueeze(torch.stack([channel2mask[i] if i in channel2mask else torch.zeros_like(result[0, 0]) for i in range(result.shape[1])]), 0).to(device)

            iou = iou_concat(result, mask)
            ious.append(iou)

        ious = np.array(ious)
        return np.mean(ious)


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded.mean() if you are interested in average across the batch


# Numpy version
# Well, it's the same function, so I'm going to omit the comments

def iou_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze(1)

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return thresholded  # Or thresholded.mean()


if __name__ == '__main__':
    device = 'cuda'
    model = Unet(layers=[8, 16, 32, 64, 128], output_channels=11).to(device)

    _, dataloader_train, _, dataloader_test = get_dataloaders_supervised()
    model.load_state_dict(torch.load('learned_models/without_transfer/without_transfer_100_1e-5.pt', map_location=device))

    dataloader_test_first = [next(iter(dataloader_train))]
    ious = iou_global(dataloader_test_first, model, device)
    print(ious)