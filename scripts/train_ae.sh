# # CIFAR10 training
# python main/train_ae.py +dataset=cifar10/train \
#                      dataset.data.root='/home/pandeyk1/datasets/' \
#                      dataset.data.name='cifar10' \
#                      dataset.data.hflip=True \
#                      dataset.model.code_size=128 \
#                      dataset.model.encoder.base_ch=64 \
#                      dataset.model.encoder.channel_mults=[1,2,2,2,2] \
#                      dataset.training.batch_size=256 \
#                      dataset.training.epochs=100 \
#                      dataset.training.device=\'gpu:6\' \
#                      dataset.training.results_dir=\'/home/pandeyk1/cs274e/cifar10/vae_cifar10_beta=0.1/\' \
#                      dataset.training.workers=1 \
#                      dataset.training.chkpt_prefix=\'cifar10_beta=0.1\' \
#                      dataset.training.beta=0.1


# # MNIST training
# python main/train_ae.py +dataset=mnist/train \
#                      dataset.data.root='/home/pandeyk1/datasets/' \
#                      dataset.data.name='mnist' \
#                      dataset.data.hflip=True \
#                      dataset.model.code_size=16 \
#                      dataset.model.encoder.base_ch=64 \
#                      dataset.model.encoder.channel_mults=[1,2,2,2,2] \
#                      dataset.training.batch_size=256 \
#                      dataset.training.epochs=100 \
#                      dataset.training.device=\'gpu:2\' \
#                      dataset.training.results_dir=\'/home/pandeyk1/cs274e/mnist/vae_cifar10_beta=100.0/\' \
#                      dataset.training.workers=1 \
#                      dataset.training.chkpt_prefix=\'cifar10_beta=100.0\' \
#                      dataset.training.beta=100.0


# Celeba training
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