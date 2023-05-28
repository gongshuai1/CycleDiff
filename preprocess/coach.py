import torch.nn as nn
import torch
import torch.utils.data
import os
import time
import torchvision.models as models
from matplotlib import pyplot as plt

plt.switch_backend('agg')
from collections import OrderedDict
from dataset.CelebaAHQDataset import (CelebAHQDataset, CelebAHQTestDataset)
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize


class Coach:
    def __init__(self, args):
        self.args = args
        self.global_step = 0

        # Load model
        self.task_num = self.args.attribute_count
        self.attribute_classifier = AttributeClassifier(self.args.attribute_count)
        self.attribute_classifier.cuda(self.args.gpu)
        self.attribute_classifier = torch.nn.parallel.DistributedDataParallel(
            self.attribute_classifier, device_ids=[self.args.gpu], broadcast_buffers=False, find_unused_parameters=True
        )

        # print(self.attribute_classifier)

        # Dataset
        self.data_loader = self.configure_dataset()
        # self.test_loader = self.configure_test_dataset()

        # Loss
        self.loss = AttributeLoss(self.args.attribute_count).cuda(self.args.gpu).eval()
        self.loss_dict = []
        # self.accuracy = []

        # Optimizer - 3000 for test
        self.optimizer = self.configure_optimizer()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer, T_max=self.args.max_steps * (30000// (self.args.batch_size * 2)))

        # Checkpoint
        self.checkpoint_dir = os.path.join(self.args.exp_dir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.args.save_interval is None:
            self.args.save_interval = self.args.max_steps

        # Logging
        self.log_dir = os.path.join(self.args.exp_dir, "log")
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=self.log_dir)

    def configure_dataset(self):
        dataset = CelebAHQDataset(self.args.data_dir)
        data_sampler = torch.utils.data.DistributedSampler(dataset)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size, sampler=data_sampler,
            collate_fn=CelebAHQDataset.collate_fn, drop_last=False
        )
        return data_loader

    def configure_test_dataset(self):
        testset = CelebAHQTestDataset(self.args.data_dir)
        test_sampler = torch.utils.data.DistributedSampler(testset)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=self.args.batch_size, sampler=test_sampler,
            collate_fn=CelebAHQTestDataset.collate_fn, drop_last=False
        )
        return test_loader

    def configure_optimizer(self):
        params = list(self.attribute_classifier.module.parameters())
        optimizer = torch.optim.Adam(params, lr=self.args.learning_rate)
        return optimizer

    def validate_testset(self):
        correct_count = 0
        for _, (x, y) in enumerate(self.test_loader):
            x = x.cuda(self.args.gpu, non_blocking=True)
            y = y.cuda(self.args.gpu, non_blocking=True)
            # inference
            y_hat = self.attribute_classifier(x)
            correct = torch.sum((torch.abs(y_hat - y) < 0.1).to(dtype=torch.int32), dim=1)
            correct_count += torch.sum((correct == 0).to(dtype=torch.int32))
        return correct_count / 3000

    def train(self):
        self.attribute_classifier.train()
        start = time.time()
        lr = []
        while self.global_step <= self.args.max_steps:
            epoch_loss = 0.0
            for batch_idx, (x, y) in enumerate(self.data_loader):
                images, attributes = x, y['y']

                images = images.cuda(self.args.gpu, non_blocking=True)
                attributes = attributes.cuda(self.args.gpu, non_blocking=True)

                # forward
                y_hat = self.attribute_classifier(images)

                # compute loss
                loss = self.loss(y=attributes, y_hat=y_hat)
                # print(f'step = {self.global_step}, batch_idx = {batch_idx}, loss = {loss}')
                epoch_loss = (epoch_loss * batch_idx + loss.detach()) / (batch_idx + 1)

                # back forward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                lr.append(self.scheduler.get_lr()[0])
                self.scheduler.step()  # Update learning rate

            # save loss
            self.loss_dict.append(epoch_loss)
            self.save_loss(self.global_step, epoch_loss)

            # Validate testset
            # self.accuracy.append(self.validate_testset())

            # checkpoint and log
            if self.global_step % self.args.save_interval == 0 or self.global_step == self.args.max_steps:
                # save checkpoint
                self.save_checkpoint()
                # plot loss
                self.plot_loss()

            end = time.time()
            print(f'Training - {self.global_step}/{self.args.max_steps}:  {end - start} seconds, loss = {epoch_loss}')

            self.global_step += 1

    def save_checkpoint(self):
        if self.args.gpu != 0:
            return

        save_path = os.path.join(self.checkpoint_dir, f'iteration_{self.global_step}.pt')
        save_dict = self.__get_save_dict()
        torch.save(save_dict, save_path)

    def save_loss(self, step, loss):
        if self.args.gpu != 0:
            return

        with open(os.path.join(self.log_dir, 'timestamp.txt'), 'a') as f:
            f.write(f'Step - {step}, Loss - {loss}\n')
            f.close()

    def plot_loss(self):
        if self.args.gpu != 0:
            return

        steps = range(len(self.loss_dict))
        loss = [item.detach().cpu() for item in self.loss_dict]
        plt.plot(steps, loss)
        plt.title('Loss-steps', fontsize=24)
        plt.xlabel('steps', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.tick_params(axis='both', labelsize=14)
        plt.savefig(os.path.join(self.log_dir, f'{self.global_step}-loss.png'))

        # accuracy = [item.detach().cpu() for item in self.accuracy]
        # plt.plot(steps, accuracy)
        # plt.title('Accuracy-steps', fontsize=24)
        # plt.xlabel('steps', fontsize=14)
        # plt.ylabel('accuracy', fontsize=14)
        # plt.tick_params(axis='both', labelsize=14)
        # plt.savefig(os.path.join(self.log_dir, f'{self.global_step}-accuracy.png'))

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.attribute_classifier.state_dict(),
            'opts': vars(self.args)
        }
        return save_dict


def normalize(inputs):
    d_min = inputs.min()
    d_max = inputs.max()
    inputs = (inputs - d_min) / (d_max - d_min)
    return inputs


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class AttributeClassifier(nn.Module):
    def __init__(self, att_num, cpk_path=None):
        super(AttributeClassifier, self).__init__()
        self.att_num = att_num

        self.resize = Resize(224)
        self.base_model = models.resnet18(pretrained=True)
        self.classify_headers = nn.ModuleList([])
        for _ in range(self.att_num):
            self.classify_headers.append(FeedForward(1000, 768))

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
            self.classify_headers.load_state_dict(cpk_classify_headers)

    def forward(self, x):
        x = self.resize(x)
        feature = self.base_model(x)
        y = [header(feature) for header in self.classify_headers]
        y = torch.squeeze(torch.stack(y, dim=0), dim=2).permute(1, 0)
        return y


class AttributeLoss(nn.Module):
    def __init__(self, att_num):
        super(AttributeLoss, self).__init__()
        self.att_num = att_num
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, y, y_hat):
        loss = self.loss_fn(y_hat, y) * self.att_num
        return loss
