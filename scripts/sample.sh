# CelebA Samples
python main/eval/sample.py +dataset=celeba64/train \
                     dataset.data.root='/home/pandeyk1/datasets/img_align_celeba' \
                     dataset.data.name='celeba' \
                     dataset.data.hflip=False \
                     dataset.model.code_size=64 \
                     dataset.model.encoder.base_ch=64 \
                     dataset.model.encoder.channel_mults=[1,2,2,2,2] \
                     dataset.inference.chkpt_path=\'/home/pandeyk1/cs274e/celeba/vae_celeba_beta=1.0/checkpoints/vae-celeba_beta=1.0-epoch=99-train_loss=0.0000.ckpt\' \
                     dataset.inference.device=\'cuda:3\' \
                     dataset.inference.save_path=\'/home/pandeyk1/cs274e/celeba/samples/beta=1.0/\' \
                     dataset.inference.write_mode='image' \
                     dataset.inference.n_samples=10 \
