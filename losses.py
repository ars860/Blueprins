import torch


def focal_loss(bce_loss, targets, gamma=2, alpha=0.25):
    """Binary focal loss, mean.

    Per https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/5 with
    improvements for alpha.
    :param bce_loss: Binary Cross Entropy loss, a torch tensor.
    :param targets: a torch tensor containing the ground truth, 0s and 1s.
    :param gamma: focal loss power parameter, a float scalar.
    :param alpha: weight of the class indicated by 1, a float scalar.
    """
    p_t = torch.exp(-bce_loss)
    alpha_tensor = (1 - alpha) + targets * (2 * alpha - 1)  # alpha if target = 1 and 1 - alpha if target = 0
    f_loss = alpha_tensor * (1 - p_t) ** gamma * bce_loss
    return f_loss.mean()


def dice_loss(pred, target, smooth=1., channels=None):
    if channels is None:
        channels = list(range(pred.shape[1]))

    # channels = [channel for channel in range(pred.shape[1]) if channel not in ignored_channels]  # list(set(range()) - set(ignored_channels))

    pred = pred[:, channels].contiguous()
    target = target[:, channels].contiguous()

    # print(pred.shape)

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()