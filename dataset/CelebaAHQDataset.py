import os
import torch.utils.data
import numpy as np
import blobfile as bf
import torchvision.transforms as transform
import torch.nn.functional as F
from PIL import Image


class CelebAHQDataset(torch.utils.data.Dataset):
    """
    CelebA dataset for facial generation and manipulate
    """
    def __init__(self, data_dir_path):
        super(CelebAHQDataset, self).__init__()
        self.data_dir_path = data_dir_path
        # Attribute for diffusion
        self.attribute_path = os.path.join(self.data_dir_path, 'attribute_20_30000.pt')
        # Attribute for classifier
        # self.attribute_path = os.path.join(self.data_dir_path, 'attribute_8_30000.pt')
        self.attribute = torch.load(self.attribute_path)

        # Count of images
        if not os.path.exists(self.data_dir_path):
            self.length = 0
        else:
            # Length for diffusion
            # self.length = len(os.listdir(self.data_dir_path)) - 4  # 4 for attribute.pt, attribute_8_30000.pt, attribute_20_30000.pt and index_mapping.txt
            # Length for classifier
            # self.length = len(os.listdir(self.data_dir_path)) - 4
            self.length = len(os.listdir(self.data_dir_path)) - 4 - 3000  # 3000 for test

        assert len(self.attribute) >= self.length

    def __getitem__(self, index):
        if self.length == 0:
            return None
        # For diffusion - 30000
        file_path = os.path.join(self.data_dir_path, f'%05d.jpg' % index)
        # For attribute classifier - 200000
        # file_path = os.path.join(self.data_dir_path, f'%06d.jpg' % (index + 1))

        # Load images
        with bf.BlobFile(file_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        x = np.array(pil_image.convert("RGB"))
        x = x.astype(np.float32) / 127.5 - 1
        x = torch.from_numpy(x)
        x = x.permute(2, 0, 1)  # (C, H, W)

        # Load attributes
        y = self.attribute[index]
        # y = (y + 1) // 2  # convert [-1, 1] to [0, 1]

        return x, y

    def __len__(self):
        return self.length

    @staticmethod
    def collate_fn(batch):
        images = []
        attributes = []
        for x, y in batch:
            images.append(x)
            attributes.append(y)
        images = torch.stack(images, dim=0)
        attributes = torch.stack(attributes, dim=0)
        # Dict
        out_dict = {'y': attributes}
        return images, out_dict


class CelebAHQTestDataset(torch.utils.data.Dataset):

    """
    CelebA dataset for facial generation and manipulate
    """
    def __init__(self, data_dir_path):
        super(CelebAHQTestDataset, self).__init__()
        self.data_dir_path = data_dir_path
        self.attribute_path = os.path.join(self.data_dir_path, 'attribute_20_30000.pt')
        self.attribute = torch.load(self.attribute_path)

        # Count of images
        self.length = 3000

    def __getitem__(self, index):
        index = index + 27000
        file_path = os.path.join(self.data_dir_path, f'%05d.jpg' % index)

        # Load images
        with bf.BlobFile(file_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        x = np.array(pil_image.convert("RGB"))
        x = x.astype(np.float32) / 127.5 - 1
        x = torch.from_numpy(x)
        x = x.permute(2, 0, 1)  # (C, H, W)

        # Load attributes
        y = self.attribute[index]
        # y = (y + 1) // 2  # convert [-1, 1] to [0, 1]

        return x, y

    def __len__(self):
        return self.length

    @staticmethod
    def collate_fn(batch):
        images = []
        attributes = []
        for x, y in batch:
            images.append(x)
            attributes.append(y)
        images = torch.stack(images, dim=0)
        attributes = torch.stack(attributes, dim=0)
        return images, attributes
