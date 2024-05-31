import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class BiRefNetImageProcessor:
    def __init__(self, size: int):
        self.image_size = (size, size)
        self.transform_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __call__(self, _image: Image.Image) -> torch.Tensor:
        _image_rs = cv2.resize(np.array(_image), self.image_size, interpolation=cv2.INTER_LINEAR)
        _image_rs = Image.fromarray(_image_rs)
        image = self.transform_image(_image_rs)
        return image
