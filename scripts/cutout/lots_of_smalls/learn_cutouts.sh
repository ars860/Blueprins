nohup python3.7 -u train.py --root "datasets/mask_cutout_100_10x10_gray_patch" --device "cuda:1" --lr "1e-4" --epochs 100 --save "cutouts/100epochs_1e-4_mask_cutout_100_10x10_gray_patch_transfer" --cutout --transfer "autoencoder/gaussian_noise/0_5_10epochs.pt" --checkpoint -1 > logs/cutouts/100epochs_1e-4_mask_cutout_100_10x10_gray_patch_transfer.log &

nohup python3.7 -u train.py --root "datasets/mask_cutout_100_10x10_gray_patch_cut_mask" --device "cuda:1" --lr "1e-4" --epochs 100 --save "cutouts/100epochs_1e-4_mask_cutout_100_10x10_gray_patch_transfer_cut_mask" --cutout --transfer "autoencoder/gaussian_noise/0_5_10epochs.pt" --checkpoint -1 > logs/cutouts/100epochs_1e-4_mask_cutout_100_10x10_gray_patch_transfer_cut_mask.log &

nohup python3.7 -u train.py --root "datasets/mask_cutout_100_10x10_gray_patch_drop_initial" --device "cuda:2" --lr "1e-4" --epochs 100 --save "cutouts/100epochs_1e-4_mask_cutout_100_10x10_gray_patch_transfer_drop_initial" --cutout --transfer "autoencoder/gaussian_noise/0_5_10epochs.pt" --checkpoint -1 > logs/cutouts/100epochs_1e-4_mask_cutout_100_10x10_gray_patch_transfer_drop_initial.log &

#Best, loss on test decreasing maybe
nohup python3.7 -u train.py --root "datasets/mask_cutout_100_10x10_gray_patch_cut_mask_drop_initial" --device "cuda:2" --lr "1e-4" --epochs 200 --save "cutouts/200epochs_1e-4_mask_cutout_100_10x10_gray_patch_transfer_cut_mask_drop_initial" --cutout --transfer "autoencoder/gaussian_noise/0_5_10epochs.pt" --checkpoint -1 > logs/cutouts/200epochs_1e-4_mask_cutout_100_10x10_gray_patch_transfer_cut_mask_drop_initial.log &
