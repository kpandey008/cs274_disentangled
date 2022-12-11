# fidelity --gpu 3 --fid --input1 ~/datasets/CelebAMask-HQ-128/ --input2 ~/ddpm_cmhq_confirm_form2/49/images/ --input1-cache-name cmhq128
# fidelity --gpu 3 --fid --input1 /home/pandeyk1/cs274e/celeba/recons/beta=1.0/orig/ --input2 /home/pandeyk1/cs274e/celeba/recons/beta=100.0/recons/
# fidelity --gpu 3 --fid --input1 /home/pandeyk1/cs274e/cifar10/recons/beta=100.0/recons/ --input2 cifar10-train
fidelity --gpu 3 --fid --input1 /home/pandeyk1/cs274e/mnist/recons/beta=0.05/orig/ --input2 /home/pandeyk1/cs274e/mnist/recons/beta=100.0/recons/