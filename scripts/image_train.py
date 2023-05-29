"""
Train a diffusion model on images.
"""

import argparse
import os

from diffusion import dist_util, logger
from diffusion.image_datasets import load_data
from diffusion.resample import create_named_schedule_sampler
from diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="/training_data/CelebA/celeba-hq-30000/celeba-256-tmp",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=500000,
        batch_size=4,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        # save_interval=100,
        # resume_checkpoint="/home/gongshuai/con-diffusion/20221012-baseline-1/model400000.pt",
        resume_checkpoint=None,
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(
        {
            # "attention_resolutions": "32, 16, 8",
            "class_cond": True,
            "diffusion_steps": 1000,
            "rescale_timesteps": True,
            # "timestep_respacing": 1000,
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
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    os.environ['OPENAI_LOGDIR'] = '/home/gongshuai/con-diffusion/20230528-attribute20/diffusion/'

    main()
