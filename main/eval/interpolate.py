import logging
import os
import sys

p = os.path.join(os.path.abspath("."), "main")
sys.path.insert(1, p)

import hydra
import numpy as np
import torch
import copy

from omegaconf import OmegaConf
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.latent import UncondLatentDataset
from models.module import VAE
from models.modelstore import get_model
from util import save_as_images, get_dataset


logger = logging.getLogger(__name__)


@hydra.main(config_path=os.path.join(p, "configs"))
def interpolate(config):
    # Get config and setup
    config = config.dataset
    logger.info(OmegaConf.to_yaml(config))

    # Set seed and configure device (Multi-GPU inference not supported)
    seed_everything(config.inference.seed, workers=True)
    dev = config.inference.device
    n_interpolations = config.inference.n_interpolations

    # Dataset
    root = config.data.root
    d_type = config.data.name
    image_size = config.data.image_size
    num_samples = config.inference.n_samples
    dataset = get_dataset(d_type, root, image_size, norm=True, flip=config.data.hflip)

    # Model
    enc, dec = get_model(config.data.name, config)
    vae = VAE.load_from_checkpoint(config.inference.chkpt_path, enc=enc, dec=dec).to(dev)
    vae.eval()

    # Sample latent codes
    x = dataset[8].unsqueeze(0).to(dev)
    dim_idx = config.inference.inter_dim

    with torch.no_grad():
        mu, logvar = vae.encode(x)
        z_1 = vae.reparameterize(mu, logvar)
    # z_1 = torch.randn(1, config.model.code_size, device=dev)
    # z_1 = torch.zeros(1, config.model.code_size, device=dev)
    # num = z_1[:, dim_idx].item()
    lam = torch.linspace(-3, 3, n_interpolations, device=dev)

    sample_list = []
    for l in lam:
        with torch.no_grad():
            # Linear interpolation
            z_inter = copy.deepcopy(z_1)
            z_inter[:, dim_idx] = l
            recons = vae.forward(z_inter)
        sample_list.append(recons.cpu())

    cat_sample = torch.cat(sample_list, dim=0)

    # Save the image and reconstructions as numpy arrays
    save_path = config.inference.save_path
    os.makedirs(save_path, exist_ok=True)

    if config.inference.write_mode == "image":
        save_as_images(
            cat_sample.squeeze(),
            file_name=os.path.join(save_path, "vae"),
            denorm=config.inference.denorm,
        )
    else:
        np.save(os.path.join(save_path, "recons.npy"), cat_sample.numpy())


if __name__ == '__main__':
    interpolate()
