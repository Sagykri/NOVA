from torchvision import transforms
import torch

class HorizontalVerticalFlipAugmentation(object):
    def __init__(self)->None:
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

    def __call__(self, image:torch.Tensor)->torch.Tensor:
        return self.transform(image)