import torch
import os
import torchvision
import blobfile as bf
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch.utils.data
import matplotlib.pyplot as plt
from diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion
)
from collections import OrderedDict
from torchvision.transforms import Resize
from einops import rearrange
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
# from preprocess.coach import AttributeClassifier


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            # nn.GELU(),
            # nn.Dropout(dropout),
            # nn.Linear(dim // 2, 1),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class ImageEncoder(nn.Module):
    def __init__(self, dim, cpk_path=None):
        super().__init__()
        self.dim = dim
        self.model = nn.ModuleList([
            Resize(224),
            models.resnet18(pretrained=True),
            nn.GELU(),
            nn.Linear(1000, self.dim),
            nn.GELU(),
            # nn.Linear(256, self.dim),
            # nn.GELU(),
        ])

        if cpk_path is not None and os.path.exists(cpk_path):
            checkpoint = torch.load(cpk_path, map_location='cpu')
            cpk_base_model = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('module.base_model.model.'):
                    cpk_base_model[k.replace('module.base_model.model.', '')] = v
            self.model.load_state_dict(cpk_base_model)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


class AttributeClassifierHeaders(nn.Module):
    def __init__(self, in_dim, att_num, cpk_path=None):
        super(AttributeClassifierHeaders, self).__init__()
        self.att_num = att_num
        self.in_dim = in_dim

        self.classify_headers = nn.ModuleList([])
        for _ in range(self.att_num):
            self.classify_headers.append(FeedForward(self.in_dim))

        if cpk_path is not None and os.path.exists(cpk_path):
            checkpoint = torch.load(cpk_path, map_location='cpu')
            cpk_classify_headers = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('module.classify_headers.classify_headers.'):
                    cpk_classify_headers[k.replace('module.classify_headers.classify_headers.', '')] = v
            self.classify_headers.load_state_dict(cpk_classify_headers)

    def forward(self, x):
        y = [header(x) for header in self.classify_headers]
        y = [y[2], y[9], y[11], y[12], y[15], y[18]]
        y = torch.squeeze(torch.stack(y, dim=0), dim=2)
        y = y.permute(1, 0, 2)
        return y


class AttributeClassifier(nn.Module):
    def __init__(self, feature_dim, att_num, cpk_path=None):
        super(AttributeClassifier, self).__init__()
        self.feature_dim = feature_dim
        self.att_num = att_num

        self.base_model = ImageEncoder(dim=self.feature_dim)
        self.classify_headers = AttributeClassifierHeaders(in_dim=self.feature_dim, att_num=self.att_num)

        if cpk_path is not None and os.path.exists(cpk_path):
            checkpoint = torch.load(cpk_path, map_location='cpu')
            cpk_base_model = OrderedDict()
            cpk_classify_headers = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if k.startswith('module.base_model.'):
                    cpk_base_model[k.replace('module.base_model.', '')] = v
                if k.startswith('module.classify_headers.'):
                    cpk_classify_headers[k.replace('module.classify_headers.', '')] = v
            # print('state_dict = ', checkpoint['state_dict'])
            self.base_model.load_state_dict(cpk_base_model)
            self.classify_headers.load_state_dict(cpk_classify_headers, strict=False)

    def forward(self, x):
        feature = self.base_model(x)
        y = self.classify_headers(feature)
        return y


class CelebAHQDataset(torch.utils.data.Dataset):
    """
    CelebA dataset for facial generation and manipulate
    """
    def __init__(self, data_dir_path):
        super(CelebAHQDataset, self).__init__()
        self.data_dir_path = data_dir_path

        # Count of images1
        if not os.path.exists(self.data_dir_path):
            self.length = 0
        else:
            # Length for diffusion
            self.length = len(os.listdir(self.data_dir_path))

    def __getitem__(self, index):
        if self.length == 0:
            return None
        # For diffusion - 30000
        file_path = os.path.join(self.data_dir_path, f'%05d.jpg' % index)

        # Load images1
        with bf.BlobFile(file_path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        x = np.array(pil_image.convert("RGB"))
        x = x.astype(np.float32) / 127.5 - 1
        x = torch.from_numpy(x)
        x = x.permute(2, 0, 1)  # (C, H, W)

        return x

    def __len__(self):
        return self.length


def load_image(file_path):
    # Load images1
    with bf.BlobFile(file_path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()

    x = np.array(pil_image.convert("RGB"))
    x = x.astype(np.float32) / 127.5 - 1
    x = torch.from_numpy(x)
    x = x.permute(2, 0, 1)  # (C, H, W)
    return x


def generate_samples_uncondition():
    attribute_path = '/training_data/CelebA/celeba-hq-30000/celeba-256-tmp/attribute_20_30000.pt'
    save_path = '/home/gongshuai/con-diffusion/20230528-attribute20/celeba/data_distribution/'
    attributes = torch.load(attribute_path)[0:1000]
    print(f'attributes.shape = {attributes.shape}')

    model_config = model_and_diffusion_defaults()
    model_config.update(
        {
            # "attention_resolutions": "32, 16, 8",
            "class_cond": True,
            "diffusion_steps": 1000,
            "rescale_timesteps": True,
            "timestep_respacing": "1000",
            "image_size": 256,
            # "learn_sigma": True,
            "learn_sigma": False,
            "noise_schedule": "linear",
            "num_channels": 128,
            "num_res_blocks": 2,
            # "resblock_updown": True,
            # "use_fp16": True,
            "use_scale_shift_norm": True,
        }
    )

    image_size = (model_config["image_size"], model_config["image_size"])

    # Load models
    device = "cuda:3"
    print("Using device:", device)

    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(
        torch.load(
            "/home/gongshuai/con-diffusion/20230528-attribute20/celeba/checkpoint/model470000.pt",
            map_location="cpu",
        )
    )
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()

    def generate_by_attributes(index, attribute):
        attribute = attribute.to(device)
        sample_func = diffusion.ddim_sample_loop_progressive
        samples = sample_func(
            model,
            (
                1,
                3,
                model_config["image_size"],
                model_config["image_size"],
            ),
            clip_denoised=False,
            model_kwargs={"y": attribute},
            cond_fn=None,
            progress=True,
            skip_timesteps=0,
            init_image=None,
            postprocess_fn=None,
            truncate=None,
        )

        total_steps = 1000 - 1
        for j, sample in enumerate(samples):
            result = sample['pred_xstart']
            if j == 99:
                file_path = os.path.join(save_path, "900", f'%05d.jpg' % index)
                torchvision.utils.save_image(result, file_path, normalize=True, scale_each=True, range=(-1, 1))
            if j == 399:
                file_path = os.path.join(save_path, "600", f'%05d.jpg' % index)
                torchvision.utils.save_image(result, file_path, normalize=True, scale_each=True, range=(-1, 1))
            if j == total_steps:
                file_path = os.path.join(save_path, "0", f'%05d.jpg' % index)
                torchvision.utils.save_image(result, file_path, normalize=True, scale_each=True, range=(-1, 1))

    for i in range(121, len(attributes)):
        generate_by_attributes(i, attributes[i:i+1])


def visualize():
    data_dir_path = '/home/gongshuai/con-diffusion/20230528-attribute20/celeba/data_distribution/origin/'
    cpk_path = '/home/gongshuai/con-diffusion/20230528-attribute20/celeba/checkpoint/iteration_250.pt'
    save_path = '/home/gongshuai/con-diffusion/20230414-attribute40/sample/analyze/sample_999_free.png'
    batch_size = 16

    # Model
    model = AttributeClassifier(feature_dim=256, att_num=20, cpk_path=cpk_path)

    # Extract feature
    dataset = CelebAHQDataset(data_dir_path)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    features = []
    for batch_id, batch in enumerate(data_loader):
        feature = model(batch)
        features.append(feature)
    features = torch.stack(features, dim=0)
    print(f'features.shape = {features.shape}')
    features = rearrange(features, 'n b a c -> (n b) a c')  # (n, 6, c)

    # PCA
    features = rearrange(features, 'n a c -> (n a) c')
    features = features.detach().numpy()
    print(f'features.shape = {features.shape}')
    pca1 = PCA(n_components=(768 // 32))
    pca2 = PCA(n_components=3)
    x = pca1.fit_transform(features)
    print(f'pca1_x.shape = {x.shape}')
    x = pca2.fit_transform(x)
    print(f'pca12_x.shape = {x.shape}')

    def normalize(x):
        mean = [0., 0., 0.]
        std = [0., 0., 0.]

        mean += x.mean(1).sum(0)
        std += x.std(1).sum(0)

        mean = mean / len(x)
        std = std / len(x)

        print(f'mean = {mean}, std = {std}')

        for i in range(3):
            x[:, i] = (x[:, i] - mean[i]) / std[i]
        return x

    x = normalize(x)

    x = rearrange(x, '(n a) c -> n a c', a=6)

    print(f'x.shape = {x.shape}')

    # Plot
    # plt.plot(x[:, 0, 0], x[:, 0, 1], 'r.')
    # plt.plot(x[:, 1, 0], x[:, 1, 1], 'g.')
    # plt.plot(x[:, 2, 0], x[:, 2, 1], 'b.')
    # plt.plot(x[:, 3, 0], x[:, 3, 1], 'y.')
    # plt.plot(x[:, 4, 0], x[:, 4, 1], 'c.')
    # plt.plot(x[:, 5, 0], x[:, 5, 1], 'k.')
    # # plt.savefig(save_path)
    # plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x[:, 0, 0], x[:, 0, 1], x[:, 0, 2], 'r', label='bags_under_eyes')
    ax.scatter(x[:, 1, 0], x[:, 1, 1], x[:, 1, 2], 'g', label='eyeglasses')
    ax.scatter(x[:, 2, 0], x[:, 2, 1], x[:, 2, 2], 'b', label='male')
    ax.scatter(x[:, 3, 0], x[:, 3, 1], x[:, 3, 2], 'y', label='open_mouth')
    ax.scatter(x[:, 4, 0], x[:, 4, 1], x[:, 4, 2], 'c', label='smile')
    ax.scatter(x[:, 5, 0], x[:, 5, 1], x[:, 5, 2], 'k', label='lipstick')
    ax.legend(loc='best')
    ax.grid(False)
    # ax.xaxis.set_ticklabels([])
    # ax.yaxis.set_ticklabels([])
    # ax.zaxis.set_ticklabels([])

    ax.axis('off')
    # plt.xticks([])
    # plt.yticks([])
    # plt.zticks([])
    # plt.savefig(save_path)
    plt.show()


def attribute_accuracy():
    """
    Metric for each attribute accuracy
    :return:
    """
    data_dir = '/home/gongshuai/con-diffusion/20230528-attribute20/celeba/all_samples/all/'
    cpk_path = '/home/gongshuai/con-diffusion/20230528-attribute20/celeba/checkpoint/iteration_250.pt'
    attribute_path = '/training_data/CelebA/celeba-hq-30000/celeba-256-tmp/attribute_20_30000.pt'
    threshold = 0.5
    # Attributes
    attributes = torch.load(attribute_path)
    # Model
    model = AttributeClassifier(256, 20, cpk_path=cpk_path)
    count = torch.zeros_like(attributes[0])
    length = 0
    for file_name in os.listdir(data_dir):
        name = file_name.split(".")[0]
        index, attribute_index = int(name.split("_")[0]), int(name.split("_")[1])
        # index = int(file_name[:5])
        image = load_image(os.path.join(data_dir, file_name))
        image = torch.unsqueeze(image, dim=0)
        attribute_pred = torch.squeeze(model(image))
        attribute_ref = attributes[index].repeat(1)
        if attribute_index < 20:
            attribute_ref[attribute_index] = 1 - attribute_ref[attribute_index]
        count += (torch.abs(attribute_ref - attribute_pred) < threshold).to(torch.int)
        length += 1
    for i in range(len(count)):
        print(f'attribute: {i}, accuracy = {count[i] / length}')


if __name__ == '__main__':
    # generate_samples_uncondition()
    visualize()
    # attribute_accuracy()
    # file_name = "12453_12.jpg"
    # name = file_name.split(".")[0]
    # index, attribute_index = name.split("_")[0], name.split("_")[1]
    # print(f'index = {index}, attribute_index = {attribute_index}')
