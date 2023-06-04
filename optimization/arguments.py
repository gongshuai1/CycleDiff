import argparse


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument(
        "-a", "--attribute", type=str, help="The prompt for the init editing",
        default="0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1"
    )
    parser.add_argument(
        "-qa", "--query_attribute", type=str, help="The prompt for the desired editing",
        default="0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1"
    )
    parser.add_argument(
        "-i", "--init_image", type=str, help="The path to the source image input",
        default="/home/gongshuai/con-diffusion/20221012-baseline-1/sample/0.jpg"
    )
    parser.add_argument(
        "--dataset_dir", type=str, help="The path to the dataset",
        default="/training_data/CelebA/celeba-hq-30000/celeba-256-tmp"
    )
    parser.add_argument(
        "--paper_sample_output_path", type=str, help="The path to the output",
        default="/home/gongshuai/con-diffusion/20230528-attribute20/celeba/"
    )

    # Diffusion
    parser.add_argument(
        "--skip_timesteps", type=int, help="How many steps to skip during the diffusion.", default=0
    )
    parser.add_argument(
        "--attribute_guided_diffusion",
        help="Indicator for using attribute guided diffusion (for baseline comparison)",
        # action="store_true",
        dest="attribute_guided_diffusion",
        default=True,
    )
    parser.add_argument(
        "--ddim",
        help="Indicator for using DDIM instead of DDPM",
        action="store_true",
        default=False,
    )

    # For more details read guided-diffusion/guided_diffusion/respace.py
    parser.add_argument(
        "--timestep_respacing",
        type=str,
        help="How to respace the intervals of the diffusion process (number between 1 and 1000).",
        # default="1000",
        default="1000",
    )
    parser.add_argument(
        "--model_output_size",
        type=int,
        help="The resolution of the outputs of the diffusion model",
        default=256,
        choices=[256, 512],
    )

    # Augmentations
    parser.add_argument("--aug_num", type=int, help="The number of augmentation", default=1)

    parser.add_argument("--att_num", type=int, help="The number of augmentation", default=20)
    parser.add_argument("--attribute_cpk_path", type=str,
                        help="The checkpoint path of attribute classifier model",
                        default="/home/gongshuai/con-diffusion/20230528-attribute20/celeba/checkpoint/iteration_250.pt")

    # Loss
    parser.add_argument(
        "--cycle_reconstruct_lambda",
        type=float,
        help="Controls how much the generated image should look like the origin image in the unchanged area",
        default=1000
        # default=0
    )
    parser.add_argument(
        "--attribute_guidance_lambda",
        type=float,
        help="Controls how much the image should look like the prompt",
        default=50
        # default=0
    )
    parser.add_argument(
        "--range_lambda",
        type=float,
        help="Controls how far out of range RGB values are allowed to be",
        default=50,
        # default=5,
    )
    parser.add_argument(
        "--lpips_sim_lambda",
        type=float,
        help="The LPIPS similarity to the input image",
        default=750,
        # default=7500,
    )
    parser.add_argument(
        "--id_lambda",
        type=float,
        help="The id loss to the input image",
        # default=1000,
        default=500,
    )
    parser.add_argument(
        "--ir_se50_weights",
        type=str,
        help="The checkpoint path of id loss",
        default="/home/gongshuai/con-diffusion/20220922-baseline/sample/model_ir_se50.pth",  # 177
        # default="/home/gongshuai/pythonProject/con-diffusion-result/20221029-mask/diffusion-mask/cpk/model_ir_se50.pth",
    )
    parser.add_argument("--truncate", type=float, help="The truncate of postprocess_fn", default=0.5,)
    parser.add_argument("--patch_size", type=int, help="The truncate of postprocess_fn", default=8,)

    # Misc
    parser.add_argument("--seed", type=int, help="The random seed", default=None)
    parser.add_argument("--gpu_id", type=int, help="The GPU ID", default=4)
    parser.add_argument("--output_path", type=str, default="/home/gongshuai/con-diffusion/20230528-attribute20/celeba/")
    parser.add_argument("-o", "--output_file", type=str, help="The filename to save, must be png", default="1.png")
    parser.add_argument("--iterations_num", type=int, help="The number of iterations", default=1)
    parser.add_argument("--batch_size", type=int,
                        help="The number number if images to sample each diffusion process", default=1)
    parser.add_argument("--vid", help="Indicator for saving the video of the diffusion process",
                        action="store_true", dest="save_video")
    parser.add_argument("--export_assets", help="Indicator for saving raw assets of the prediction",
                        action="store_true", dest="export_assets")

    args = parser.parse_args()
    return args
