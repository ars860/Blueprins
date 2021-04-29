nohup python3.7 -u train.py --root "datasets/mask_cutout_100_10x10_gray_patch_cut_mask_drop_initial" --device "cuda:2" --lr "1e-4" --epochs 100 --save "cutouts/100epochs_1e-4_transfer_mask_cutout_100_10x10_gray_patch_cut_mask_drop_initial" --cutout --transfer "autoencoder/gaussian_noise/0_5_10epochs.pt" --checkpoint -1 > logs/cutouts/100epochs_1e-4_transfer_mask_cutout_100_10x10_gray_patch_cut_mask_drop_initial.log &

nohup python3.7 -u train.py --root "datasets/mask_cutout_1_100x100_patch_cut_mask_drop_initial" --device "cuda:2" --lr "1e-4" --epochs 100 --save "cutouts/100epochs_1e-4_transfer_mask_cutout_1_100x100_patch_cut_mask_drop_initial" --cutout --transfer "autoencoder/gaussian_noise/0_5_10epochs.pt" --checkpoint -1 > logs/cutouts/100epochs_1e-4_transfer_mask_cutout_1_100x100_patch_cut_mask_drop_initial.log &

nohup python3.7 -u train.py --root "datasets/mask_cutout_1_100x100_gray_patch_cut_mask_drop_initial" --device "cuda:2" --lr "1e-4" --epochs 100 --save "cutouts/100epochs_1e-4_transfer_mask_cutout_1_100x100_gray_patch_cut_mask_drop_initial" --cutout --transfer "autoencoder/gaussian_noise/0_5_10epochs.pt" --checkpoint -1 > logs/cutouts/100epochs_1e-4_transfer_mask_cutout_1_100x100_gray_patch_cut_mask_drop_initial.log &

nohup python3.7 -u train.py --root "datasets/mask_cutout_100_10x10_patch_cut_mask_drop_initial" --device "cuda:2" --lr "1e-4" --epochs 100 --save "cutouts/100epochs_1e-4_transfer_mask_cutout_100_10x10_patch_cut_mask_drop_initial" --cutout --transfer "autoencoder/gaussian_noise/0_5_10epochs.pt" --checkpoint -1 > logs/cutouts/100epochs_1e-4_transfer_mask_cutout_100_10x10_patch_cut_mask_drop_initial.log &

nohup python3.7 -u train.py --root "datasets/mask_cutout_100_10x10_gray_patch_x5_cut_mask_drop_initial" --device "cuda:2" --lr "1e-4" --epochs 100 --save "cutouts/100epochs_1e-4_transfer_mask_cutout_100_10x10_gray_patch_x5_cut_mask_drop_initial" --cutout --transfer "autoencoder/gaussian_noise/0_5_10epochs.pt" --checkpoint -1 > logs/cutouts/100epochs_1e-4_transfer_mask_cutout_100_10x10_gray_patch_x5_cut_mask_drop_initial.log &

nohup python3.7 -u train.py --root "datasets/mask_cutout_100_10x10_black_patch_x5_cut_mask_drop_initial" --device "cuda:1" --lr "1e-4" --epochs 100 --save "cutouts/100epochs_1e-4_transfer_mask_cutout_100_10x10_black_patch_x5_cut_mask_drop_initial" --cutout --transfer "autoencoder/gaussian_noise/0_5_10epochs.pt" --checkpoint -1 > logs/cutouts/100epochs_1e-4_transfer_mask_cutout_100_10x10_black_patch_x5_cut_mask_drop_initial.log &
