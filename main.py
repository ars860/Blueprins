import torch

from unet import Unet
from dataset import get_dataloaders_supervised


if __name__ == '__main__':
    model = Unet(layers=[32, 64, 128, 256]).to('cuda')
    dataset_train, dataloader_train, dataset_test, dataloader_test = get_dataloaders_supervised()

    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(1.0)

    img, mask = next(iter(dataloader_train))
    img, mask = img.to('cuda'), mask.to('cuda')

    print(img.shape)
    print(model(img).shape)
