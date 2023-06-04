import os
import torch
import numpy as np
import random
import torchvision
import blobfile as bf
from pathlib import Path
from diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion
)
from einops import rearrange
from utils.data_util import load_image
from utils.MetricsAccumulator import MetricsAccumulator
from utils.visualization import (show_edited_image, save_video)
from optimization.loss import range_loss
from torchvision.transforms import functional as TF
from optimization.loss import (LpipsLoss, IDLoss, AttributeLoss, cycle_reconstruct_loss)
from preprocess.coach import (AttributeClassifier, )
from preprocess.coach import AttributeLoss as AttLoss
from matplotlib import pyplot as plt
from PIL import Image


class AttributeEditor:
    def __init__(self, args) -> None:
        self.args = args

        self.model_config = model_and_diffusion_defaults()
        self.model_config.update(
            {
                # "attention_resolutions": "32, 16, 8",
                "class_cond": True,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": self.args.timestep_respacing,
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

        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        os.makedirs(self.args.output_path, exist_ok=True)

        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

        # Load models
        self.device = torch.device(
            f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        print("Using device:", self.device)

        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        self.model.load_state_dict(
            torch.load(
                "/home/gongshuai/con-diffusion/20230528-attribute20/celeba/checkpoint/model450000.pt"
                if self.args.model_output_size == 256
                else "/home/gongshuai/con-diffusion/20230528-attribute20/celeba/checkpoint/model450000.pt",
                map_location="cpu",
            )
        )
        self.model.requires_grad_(False).eval().to(self.device)
        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()

        # Cycle reconstruct loss
        self.cycle_reconstruct_loss = cycle_reconstruct_loss

        # Attribute classifier
        self.attribute_model = AttributeClassifier(
            feature_dim=256, att_num=self.args.att_num, cpk_path=self.args.attribute_cpk_path).requires_grad_(False).eval().to(
            self.device)
        self.attribute_loss = AttributeLoss(
            attribute_classifier=self.attribute_model, loss_fn=AttLoss(att_num=self.args.att_num)).eval().to(
            self.device)

        # LPIPS
        self.lpips_model = LpipsLoss().requires_grad_(False).eval().to(self.device)

        # IR_SE - ID loss
        self.id_model = IDLoss(self.args).requires_grad_(False).eval().to(self.device)

        self.metrics_accumulator = MetricsAccumulator()

        # init
        self.index = 9881
        self.attribute_index = 12
        self.init_image, self.attribute = self.load_image(self.args.dataset_dir, self.index)
        self.query_attribute = self.attribute.clone().detach()
        if self.attribute_index < 20:
            self.query_attribute[self.attribute_index] = 1 - self.query_attribute[self.attribute_index]
        self.init_image = torch.unsqueeze(self.init_image, dim=0).to(self.device)
        self.attribute = torch.unsqueeze(self.attribute, dim=0).to(self.device)
        self.query_attribute = torch.unsqueeze(self.query_attribute, dim=0).to(self.device)
        self.attribute_mask = torch.abs(self.query_attribute - self.attribute)

    def load_image(self, data_dir, index):
        attribute_path = os.path.join(data_dir, 'attribute_20_30000.pt')
        attributes = torch.load(attribute_path)
        file_name = os.path.join(data_dir, f'%05d.jpg' % index)

        # Load images
        with bf.BlobFile(file_name, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        x = np.array(pil_image.convert("RGB"))
        x = x.astype(np.float32) / 127.5 - 1
        x = torch.from_numpy(x)
        x = x.permute(2, 0, 1)  # (C, H, W)

        # Load attributes
        y = attributes[index]
        return x, y

    def save_sample(self, img, output_dir, index, attribute_index):
        file_path = os.path.join(output_dir, f'%05d_{attribute_index}.jpg' % index)
        torchvision.utils.save_image(img, file_path, normalize=True, scale_each=True, range=(-1, 1))

    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()
        return unscaled_timestep

    def cycle_reconstrct(self, x_start, t):
        # Diffusion
        x_t_hat = self.diffusion.q_sample(x_start, t)
        # Denoising diffusion
        out = self.diffusion.p_mean_variance(
            self.model, x_t_hat, t, clip_denoised=False, model_kwargs={"y": self.attribute}
        )
        fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
        # result = out["pred_xstart"] * fac + x_t_hat * (1 - fac)
        return out["pred_xstart"]
        # return result

    def cycle_reconstrct_each_step(self, x_start, t):
        # Diffusion
        x_t_hat = self.diffusion.q_sample(x_start, t)
        # Denoising diffusion for one step
        out = self.diffusion.p_sample(
            self.model, x_t_hat, t, clip_denoised=False, model_kwargs={"y": self.attribute}
        )
        x_t_prev_hat = out["sample"]
        return x_t_prev_hat

    def edit_image_by_attribute(self):

        def cond_fn(x, t, y=None):
            with torch.enable_grad():
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t)

                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )

                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
                x_in = out["pred_xstart"] * fac + x * (1 - fac)
                # x_in = out["pred_xstart"]

                loss = torch.zeros((x.shape[0]), device=self.device)
                if self.args.cycle_reconstruct_lambda != 0:
                    x_reconstruct = self.cycle_reconstrct(x_in, t)
                    cycle_consistency_loss = self.cycle_reconstruct_loss(x_reconstruct,
                                                                         self.init_image) * self.args.cycle_reconstruct_lambda
                    loss = loss + cycle_consistency_loss
                    self.metrics_accumulator.update_metric("cycle_consistency_loss", cycle_consistency_loss.item())
                if self.args.attribute_guidance_lambda != 0:
                    attribute_loss = self.attribute_loss(x_in, y,
                                                         self.attribute_mask) * self.args.attribute_guidance_lambda
                    loss = loss + attribute_loss
                    self.metrics_accumulator.update_metric("attribute_loss", attribute_loss.item())

                if self.args.range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                    loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())

                if self.args.lpips_sim_lambda:
                    lpips_loss = self.lpips_model(x_in, self.init_image).sum() * self.args.lpips_sim_lambda
                    loss = loss + lpips_loss
                    self.metrics_accumulator.update_metric("lpips_loss", lpips_loss.item())

                if self.args.id_lambda:
                    id_loss = self.id_model(x_in, self.init_image).sum() * self.args.id_lambda
                    loss = loss + id_loss
                    self.metrics_accumulator.update_metric("id_loss", id_loss.item())

                return -torch.autograd.grad(loss, x)[0]

        for iteration_number in range(self.args.iterations_num):
            print(f"Start iterations {iteration_number}")

            sample_func = (
                self.diffusion.ddim_sample_loop_progressive
                if self.args.ddim
                else self.diffusion.p_sample_loop_progressive
            )
            samples = sample_func(
                self.model,
                (
                    self.init_image.shape[0],
                    3,
                    self.model_config["image_size"],
                    self.model_config["image_size"],
                ),
                clip_denoised=False,
                model_kwargs={"y": self.query_attribute},
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=self.args.skip_timesteps,
                init_image=self.init_image,
                postprocess_fn=None,
                truncate=self.args.truncate,
            )

            total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1
            for j, sample in enumerate(samples):
                if j == total_steps:
                    # return sample['pred_xstart']
                    result = sample['pred_xstart']
                    self.save_sample(result, self.args.paper_sample_output_path, self.index, self.attribute_index)
                    # self.save_sample(result, self.args.output_path, self.index, self.attribute_index)

    def sample_batch(self):
        total = 1000
        def random_index(count):
            indexes = []
            counter = 0
            while counter < count:
                index = random.randint(0, 30000 - 1)  # [a, b]
                if index not in indexes:
                    indexes.append(index)
                    counter += 1
            return indexes

        attribute_indexes = [40]  # gpu 0
        # attribute_indexes = [15]  # gpu 1
        # attribute_indexes = [40]  # gpu 2
        # attribute_indexes = [21]  # gpu 3
        # attribute_indexes = [31, 36]  # gpu 4
        # attribute_indexes = [39, 40]  # gpu 5
        for attr_index in attribute_indexes:
            ins = random_index(total)
            self.attribute_index = attr_index
            for i in ins:
                self.index = i
                self.init_image, self.attribute = self.load_image(self.args.dataset_dir, self.index)
                self.query_attribute = self.attribute.clone().detach()
                if self.attribute_index < 40:
                    self.query_attribute[self.attribute_index] = 1 - self.query_attribute[self.attribute_index]
                self.init_image = torch.unsqueeze(self.init_image, dim=0).to(self.device)
                self.attribute = torch.unsqueeze(self.attribute, dim=0).to(self.device)
                self.query_attribute = torch.unsqueeze(self.query_attribute, dim=0).to(self.device)
                self.attribute_mask = torch.abs(self.query_attribute - self.attribute)

                self.edit_image_by_attribute()
