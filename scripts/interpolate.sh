# python main/eval/interpolate.py +dataset=mnist/train \
#                      dataset.data.root='/home/pandeyk1/datasets/' \
#                      dataset.data.name='mnist' \
#                      dataset.data.hflip=False \
#                      dataset.model.code_size=16 \
#                      dataset.model.encoder.base_ch=64 \
#                      dataset.model.encoder.channel_mults=[1,2,2,2,2] \
#                      dataset.inference.chkpt_path=\'/home/pandeyk1/cs274e/mnist/vae_cifar10_beta=0.1/checkpoints/vae-cifar10_beta=0.1-epoch=99-train_loss=0.0000.ckpt\' \
#                      dataset.inference.device=\'cuda:3\' \
#                      dataset.inference.save_path=\'/home/pandeyk1/cs274e/mnist/interp_1/beta=0.1/\' \
#                      dataset.inference.write_mode='image'
#                      dataset.inference.inter_dim=0


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
