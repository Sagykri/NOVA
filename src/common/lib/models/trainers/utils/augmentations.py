from typing import List
from torchvision import transforms
import torch
import numpy as np

class RandomRotation(transforms.RandomRotation):
    """Random rotation for a given close set of angles
    """
    def __init__(self, angles:set[float]):
        super().__init__(0)
        self.angles = np.unique(list(angles))
    
    def forward(self, img:torch.Tensor)->torch.Tensor:
        # Randomly choose an angle from the specified set
        angle = int(np.random.choice(self.angles))
        return transforms.functional.rotate(img, angle)
    
class RotationsAndFlipsAugmentation(object):
    def __init__(self)->None:
        self.transform = transforms.Compose([
            RandomRotation({0,90,180,270}),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

    def __call__(self, image:torch.Tensor)->torch.Tensor:
        """Apply randomly rotation, horizotnal flip and vertical flip on the given image

        Args:
            image (torch.Tensor): The image to transform

        Returns:
            torch.Tensor: The transformed image
        """
        return self.transform(image)