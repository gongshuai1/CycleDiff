import blobfile as bf
import numpy as np
import torch
import os
from PIL import Image


def load_image(image_path):
    with bf.BlobFile(image_path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()

    x = np.array(pil_image.convert("RGB"))
    x = x.astype(np.float32) / 127.5 - 1
    x = torch.from_numpy(x)
    x = x.permute(2, 0, 1)  # (C, H, W)
    return x


def load_image_and_attribute(data_dir, index):
    image_path = os.path.join(data_dir, f'%05d.jpg' % index)
    attribute_path = os.path.join(data_dir, 'attribute.pt')

    assert os.path.exists(image_path), f'{image_path} not exists'
    assert os.path.exists(attribute_path), f'{attribute_path} not exists'

    image = load_image(image_path)
    attributes = torch.load(attribute_path)
    attribute = (attributes[index] + 1) // 2  # convert [-1, 1] to [0, 1]
    return image, attribute
