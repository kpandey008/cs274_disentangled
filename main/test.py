import logging
import os

import click
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.latent import UncondLatentDataset
from models.module import VAE
from models.modelstore import get_model
from util import configure_device, get_dataset, save_as_images


logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


def compare_samples(gen, refined, save_path=None, figsize=(6, 3)):
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax[0].imshow(gen.permute(1, 2, 0))
    ax[0].set_title("VAE Reconstruction")
    ax[0].axis("off")

    ax[1].imshow(refined.permute(1, 2, 0))
    ax[1].set_title("Refined Image")
    ax[1].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


def plot_interpolations(interpolations, save_path=None, figsize=(10, 5)):
    N = len(interpolations)
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=1, ncols=N, figsize=figsize)

    for i, inter in enumerate(interpolations):
        ax[i].imshow(inter.permute(1, 2, 0))
        ax[i].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


def compare_interpolations(
    interpolations_1, interpolations_2, save_path=None, figsize=(10, 2)
):
    assert len(interpolations_1) == len(interpolations_2)
    N = len(interpolations_1)
    # Plot all the quantities
    fig, ax = plt.subplots(nrows=2, ncols=N, figsize=figsize)

    for i, (inter_1, inter_2) in enumerate(zip(interpolations_1, interpolations_2)):
        ax[0, i].imshow(inter_1.permute(1, 2, 0))
        ax[0, i].axis("off")

        ax[1, i].imshow(inter_2.permute(1, 2, 0))
        ax[1, i].axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)


@cli.command()
@click.argument("z-dim", type=int)
@click.argument("chkpt-path")
@click.option("--seed", default=0, type=int)
@click.option("--device", default="gpu:1")
@click.option("--image-size", default=128)
@click.option("--num-samples", default=-1)
@click.option("--save-path", default=os.getcwd())
@click.option("--write-mode", default="image", type=click.Choice(["numpy", "image"]))
def sample(
    z_dim,
    chkpt_path,
    seed=0,
    device="gpu:0",
    image_size=128,
    num_samples=1,
    save_path=os.getcwd(),
    write_mode="image",
):
    seed_everything(seed)
    dev, _ = configure_device(device)
    if num_samples <= 0:
        raise ValueError(f"`--num-samples` can take values > 0")

    dataset = UncondLatentDataset((num_samples, z_dim, 1, 1))

    # Loader
    loader = DataLoader(
        dataset,
        16,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    vae = VAE.load_from_checkpoint(chkpt_path, input_res=image_size).to(dev)
    vae.eval()

    sample_list = []
    count = 0
    for _, batch in tqdm(enumerate(loader)):
        batch = batch.to(dev)
        with torch.no_grad():
            recons = vae.forward(batch)

        if count + recons.size(0) >= num_samples and num_samples != -1:
            sample_list.append(recons[:num_samples, :, :, :].cpu())
            break

        # Not transferring to CPU leads to memory overflow in GPU!
        sample_list.append(recons.cpu())
        count += recons.size(0)

    cat_sample = torch.cat(sample_list, dim=0)

    # Save the image and reconstructions as numpy arrays
    os.makedirs(save_path, exist_ok=True)

    if write_mode == "image":
        save_as_images(
            cat_sample,
            file_name=os.path.join(save_path, "vae"),
            denorm=False,
        )
    else:
        np.save(os.path.join(save_path, "recons.npy"), cat_sample.numpy())


if __name__ == "__main__":
    cli()
