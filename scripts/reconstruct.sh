# # CIFAR10 Reconstructions
# python main/eval/reconstruct.py +dataset=cifar10/train \
#                      dataset.data.root='/home/pandeyk1/datasets/' \
#                      dataset.data.name='cifar10' \
#                      dataset.data.hflip=False \
#                      dataset.model.code_size=128 \
#                      dataset.model.encoder.base_ch=64 \
#                      dataset.model.encoder.channel_mults=[1,2,2,2,2] \
#                      dataset.inference.chkpt_path=\'/home/pandeyk1/cs274e/cifar10/vae_cifar10_beta=0.05/checkpoints/vae-cifar10_beta=0.05-epoch=99-train_loss=0.0000.ckpt\' \
#                      dataset.inference.device=\'cuda:3\' \
#                      dataset.inference.save_path=\'/home/pandeyk1/cs274e/cifar10/recons_quals/beta=0.05/\' \
#                      dataset.inference.write_mode='image' \
#                      dataset.inference.n_samples=20 \


# python main/eval/reconstruct.py +dataset=celeba64/train \
#                      dataset.data.root='/home/pandeyk1/datasets/img_align_celeba' \
#                      dataset.data.name='celeba' \
#                      dataset.data.hflip=False \
#                      dataset.model.code_size=64 \
#                      dataset.model.encoder.base_ch=64 \
#                      dataset.model.encoder.channel_mults=[1,2,2,2,2] \
#                      dataset.inference.chkpt_path=\'/home/pandeyk1/cs274e/celeba/vae_celeba_beta=0.05/checkpoints/vae-celeba_beta=0.05-epoch=99-train_loss=0.0000.ckpt\' \
#                      dataset.inference.device=\'cuda:3\' \
#                      dataset.inference.save_path=\'/home/pandeyk1/cs274e/celeba/recons_quals/beta=0.05/\' \
#                      dataset.inference.write_mode='image' \
#                      dataset.inference.n_samples=20

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