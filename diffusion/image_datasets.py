from torch.utils.data import DataLoader
from dataset.CelebaAHQDataset import CelebAHQDataset


def load_data(
    *, data_dir, batch_size, deterministic=False
):
    """
    For a dataset, create a generator over (images1, attributes) pairs.

    Each images1 is an NCHW float tensor, and attributes indicate the features encoded by Resnet50 of each images1

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param deterministic: if True, yield results in a deterministic order.
    :param image_encoder_ckpt_path: the checkpoint path of learned image encoder
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    dataset = CelebAHQDataset(data_dir)
    if deterministic:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=CelebAHQDataset.collate_fn,
            shuffle=False,
            num_workers=1,
            drop_last=True
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=CelebAHQDataset.collate_fn,
            shuffle=True,
            num_workers=1,
            drop_last=True
        )
    while True:
        yield from loader
