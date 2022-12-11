# Exploring Disentanglement in Deep Generative Latent Variable Models using $\beta$-VAEs
An implementation of the beta-vae framework for CS274E. Our code used PyTorch Lightning + Hydra as primary dependencies which will needed to be setup to run training and inference commands. 

Some useful commands for training and inference are as follows:

## Training (for example on the CelebA-64 dataset)

```
python main/train_ae.py +dataset=celeba64/train \
                     dataset.data.root='/home/pandeyk1/datasets/img_align_celeba' \
                     dataset.data.name='celeba' \
                     dataset.data.hflip=True \
                     dataset.data.subsample_size=50000 \
                     dataset.model.code_size=64 \
                     dataset.model.encoder.base_ch=64 \
                     dataset.model.encoder.channel_mults=[1,2,2,2,2] \
                     dataset.training.batch_size=128 \
                     dataset.training.epochs=100 \
                     dataset.training.device=\'gpu:2\' \
                     dataset.training.results_dir=\'/home/pandeyk1/cs274e/celeba/vae_celeba_beta=0.05/\' \
                     dataset.training.workers=1 \
                     dataset.training.chkpt_prefix=\'celeba_beta=0.05\' \
                     dataset.training.beta=0.05
```
More training scripts can be found in the file `scripts/train_ae.sh`. The commands also specify the exact hyperparameters used for training our VAE models.

## Generating Reconstructions

Sample command for the MNIST dataset
```
python main/eval/reconstruct.py +dataset=mnist/train \
                     dataset.data.root='/home/pandeyk1/datasets/' \
                     dataset.data.name='mnist' \
                     dataset.data.hflip=False \
                     dataset.model.code_size=16 \
                     dataset.model.encoder.base_ch=64 \
                     dataset.model.encoder.channel_mults=[1,2,2,2,2] \
                     dataset.inference.chkpt_path=\'/home/pandeyk1/cs274e/mnist/vae_cifar10_beta=0.05/checkpoints/vae-cifar10_beta=0.05-epoch=99-train_loss=0.0000.ckpt\' \
                     dataset.inference.device=\'cuda:3\' \
                     dataset.inference.save_path=\'/home/pandeyk1/cs274e/mnist/recons_quals/beta=0.05/\' \
                     dataset.inference.write_mode='image' \
                     dataset.inference.n_samples=20 \
```


## Generating Interpolations

Sample command for the CelebA-64 dataset
```
python main/eval/interpolate.py +dataset=celeba64/train \
                     dataset.data.root='/home/pandeyk1/datasets/img_align_celeba/' \
                     dataset.data.name='celeba' \
                     dataset.data.hflip=False \
                     dataset.model.code_size=64 \
                     dataset.model.encoder.base_ch=64 \
                     dataset.model.encoder.channel_mults=[1,2,2,2,2] \
                     dataset.inference.chkpt_path=\'/home/pandeyk1/cs274e/celeba/vae_celeba_beta=10.0/checkpoints/vae-celeba_beta=10.0-epoch=99-train_loss=0.0000.ckpt\' \
                     dataset.inference.device=\'cuda:3\' \
                     dataset.inference.save_path=\'/home/pandeyk1/cs274e/celeba/interp/beta=10.0/\' \
                     dataset.inference.write_mode='image'
                     dataset.inference.inter_dim=0
```