import logging
import os
import sys

p = os.path.join(os.path.abspath("."), "main")
sys.path.insert(1, p)

import hydra
import numpy as np
import torch

from omegaconf import OmegaConf
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.module import VAE
from models.modelstore import get_model
from util import get_dataset, save_as_images


logger = logging.getLogger(__name__)


@hydra.main(config_path=os.path.join(p, "configs"))
def reconstruct(config):
    # Get config and setup
    config = config.dataset
    logger.info(OmegaConf.to_yaml(config))

    # Set seed and configure device (Multi-GPU inference not supported)
    seed_everything(config.inference.seed, workers=True)
    dev = config.inference.device

    # Dataset
    root = config.data.root
    d_type = config.data.name
    image_size = config.data.image_size
    num_samples = config.inference.n_samples
    dataset = get_dataset(d_type, root, image_size, norm=True, flip=config.data.hflip)

    # Loader
    loader = DataLoader(
        dataset,
        512,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    # Model
    enc, dec = get_model(d_type, config)
    vae = VAE.load_from_checkpoint(config.inference.chkpt_path, enc=enc, dec=dec).to(dev)
    vae.eval()

    sample_list = []
    img_list = []
    count = 0
    for _, batch in tqdm(enumerate(loader)):
        batch = batch.to(dev)
        with torch.no_grad():
            recons = vae.forward_recons(batch)

        if count + recons.size(0) >= num_samples and num_samples != -1:
            img_list.append(batch[:num_samples, :, :, :].cpu())
            sample_list.append(recons[:num_samples, :, :, :].cpu())
            break

        # Not transferring to CPU leads to memory overflow in GPU!
        sample_list.append(recons.cpu())
        img_list.append(batch.cpu())
        count += recons.size(0)

    cat_img = torch.cat(img_list, dim=0)
    cat_sample = torch.cat(sample_list, dim=0)

    # Save the image and reconstructions as numpy arrays
    save_path = config.inference.save_path
    orig_save_path = os.path.join(save_path, 'orig')
    recons_save_path = os.path.join(save_path, 'recons')
    os.makedirs(orig_save_path, exist_ok=True)
    os.makedirs(recons_save_path, exist_ok=True)

    if config.inference.write_mode == "image":
        save_as_images(
            cat_sample.squeeze(),
            file_name=os.path.join(recons_save_path, "vae"),
            denorm=config.inference.denorm,
        )
        save_as_images(
            cat_img.squeeze(),
            file_name=os.path.join(orig_save_path, "orig"),
            denorm=config.inference.denorm,
        )
    else:
        np.save(os.path.join(save_path, "images.npy"), cat_img.numpy())
        np.save(os.path.join(save_path, "recons.npy"), cat_sample.numpy())


if __name__ == '__main__':
    reconstruct()
