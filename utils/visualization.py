import torch
import torchvision
import numpy as np
import cv2
from typing import List
from PIL.Image import Image


def show_edited_image(
    source_image: torch.Tensor,
    edited_image: torch.Tensor,
    sample_out_path: str,
):
    torchvision.utils.save_image(torch.cat([source_image, edited_image]), sample_out_path,
                                 normalize=True, scale_each=True, range=(-1, 1))


def save_video(images_list: List[Image], video_path: str):
    """Saves a video from a list of images

    Args:
        images_list (List[Image]): A list of PIL images.
        video_path (str): The path to save to video to.
    """
    images = [np.array(img) for img in images_list]
    height, width, _ = images[0].shape

    fps = len(images) // 20
    video = cv2.VideoWriter(video_path, 0, fps, (width, height))

    for img in images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    cv2.destroyAllWindows()
    video.release()
