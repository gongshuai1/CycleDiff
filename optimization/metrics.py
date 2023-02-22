import torch
import numpy as np
import torch.nn as nn
import os
from collections import OrderedDict
from torchvision.transforms import Resize
import torchvision.models as models
import blobfile as bf
from PIL import Image
from preprocess.coach import AttributeClassifier


def load_image(file_path):
    # Load images
    with bf.BlobFile(file_path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()

    x = np.array(pil_image.convert("RGB"))
    x = x.astype(np.float32) / 127.5 - 1
    x = torch.from_numpy(x)
    x = x.permute(2, 0, 1)  # (C, H, W)
    return x


def attribute_7_filter(attribute, attribute_index=None):
    if attribute_index is not None:
        attribute[attribute_index] = 1 - attribute[attribute_index]

    attr = []
    attr.append(attribute[2])
    attr.append(attribute[9])
    attr.append(attribute[11])
    attr.append(attribute[12])
    attr.append(attribute[15])
    attr.append(attribute[18])
    attr.append(attribute[19])
    attr = torch.stack(attr, dim=0)
    return attr


def attribute_accuracy():
    """
    Metric for each attribute accuracy
    :return:
    """
    data_dir = '/home/gongshuai/con-diffusion/20221012-baseline-1/sample/attribute_classifier/all_sample/'
    cpk_path = '/home/gongshuai/con-diffusion/20221108-attribute20-resnet18/checkpoint/iteration_250.pt'
    attribute_path = '/training_data/CelebA/celeba-hq-30000/celeba-256-tmp/attribute_20_30000.pt'
    attribute_index = 2
    threshold = 0.5
    # Attributes
    attributes = torch.load(attribute_path)
    # Model
    model = AttributeClassifier(20, cpk_path=cpk_path)
    count = 0
    for file_name in os.listdir(os.path.join(data_dir, str(attribute_index))):
        index = int(file_name[:5])
        image = load_image(os.path.join(data_dir, str(attribute_index), file_name))
        image = torch.unsqueeze(image, dim=0)
        attribute_pred = attribute_7_filter(torch.squeeze(model(image)))
        attribute_ref = attribute_7_filter(attributes[index], attribute_index)
        count += torch.sum(torch.abs(attribute_ref - attribute_pred) < threshold)
    print(f'attribute: {attribute_index}, accuracy = {count / 7000}')


def reconstruction_error():
    """
    Metric for reconstruction error
    :return:
    """
    data_dir = '/home/gongshuai/con-diffusion/20221012-baseline-1/sample/attribute_classifier/all_sample/20'
    origin_dir = '/training_data/CelebA/celeba-hq-30000/celeba-256-tmp'
    attribute_path = os.path.join(origin_dir, '/attribute_20_30000.pt')
    reconstruction_error = 0.0
    files = os.listdir(data_dir)
    for file_name in os.listdir(data_dir):
        index = int(file_name[:5])
        image_recon = load_image(os.path.join(data_dir, file_name))
        image_origin = load_image(os.path.join(origin_dir, f'%05d.jpg' % index))
        # reconstruction_error += torch.mean(torch.abs(image_recon - image_origin))
        reconstruction_error += torch.dist(image_recon, image_origin, p=1)
    print(f'reconstruction_error: {reconstruction_error / len(files)}')


if __name__ == '__main__':
    attribute_accuracy()
    # reconstruction_error()
