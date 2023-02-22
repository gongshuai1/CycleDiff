import torch.nn as nn
import torch
from .models.facial_recognition.model_irse import Backbone
from .models.lpips.model_lpips import get_network, LinLayers
from .models.lpips.helpers import get_state_dict


# Cycle reconstruct loss
def cycle_reconstruct_loss(reconstructed_x, x):
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(reconstructed_x, x)
    return loss


# Attribute classifier loss
class AttributeLoss(nn.Module):
    @staticmethod
    def __data_convertor__(images, attributes):
        """
        Convert multi attribute prediction for one sample to one attribute classifier for multi samples
        using one-hot task code
        :param images: a batch of samples, [b, c, h, w]
        :param attributes: attributes, [b, 20] which '20' denotes the number of attributes
        :return: images - [b * 20, c, h, w]
                 task - [b * 20, 20]
                 attributes - [b * 20, 1]
        """
        b, _, _, _ = images.shape
        task_num = attributes.shape[1]
        images = images.repeat(task_num, 1, 1, 1)
        task = torch.zeros((b * task_num, task_num)).to(images.device)
        binary_attributes = torch.zeros((b * task_num, 1)).to(images.device)

        for i in range(task_num):
            task[i * b: (i + 1) * b, i] = 1
            for j in range(b):
                binary_attributes[i * b + j] = attributes[j, i]
        return images, task, binary_attributes

    def __init__(self, attribute_classifier, loss_fn):
        super().__init__()
        self.attribute_classifier = attribute_classifier
        self.loss_fn = loss_fn
        # self.augmentation_number = augmentations_number
        # self.image_augmentations = ImageAugmentations(output_size, augmentations_number)
        # self.celeba_normalize = transforms.Normalize(
        #     mean=[0.03367958, -0.1657348, -0.27304852], std=[0.6030259, 0.54680276, 0.5359615]
        # )

    def forward(self, images, attributes, mask):
        y_hat = self.attribute_classifier(images)
        # Only calculate loss of the changed attribute
        y_hat = mask * y_hat + (1 - mask) * attributes
        loss = self.loss_fn(y=attributes, y_hat=y_hat)
        return loss


# Range loss
def range_loss(x):
    return (x - x.clamp(-1, 1)).pow(2).mean([1, 2, 3])


# Lpips loss
class LpipsLoss(nn.Module):
    r"""Creates a criterion that measures
        Learned Perceptual Image Patch Similarity (LPIPS).
        Arguments:
            net_type (str): the network type to compare the features:
                            'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
            version (str): the version of LPIPS. Default: 0.1.
        """

    def __init__(self, net_type: str = 'alex', version: str = '0.1'):
        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LpipsLoss, self).__init__()

        # pretrained network
        # self.net = get_network(net_type).to("cuda")
        self.net = get_network(net_type)

        # linear layers
        # self.lin = LinLayers(self.net.n_channels_list).to("cuda")
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0)) / x.shape[0]


# IR_SE - ID loss
class IDLoss(nn.Module):
    def __init__(self, opts):
        super(IDLoss, self).__init__()
        self.opts = opts
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(opts.ir_se50_weights, map_location='cpu'))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        for module in [self.facenet, self.face_pool]:
            for param in module.parameters():
                param.requires_grad = False

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count
