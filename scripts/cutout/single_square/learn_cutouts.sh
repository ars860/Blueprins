nohup python3.7 -u train.py --root "datasets/mask_cutout_1_100x100_patch" --device "cuda:1" --lr "1e-4" --epochs 100 --save "cutouts/100epochs_1e-4_mask_cutout_1_100x100_patch" --cutout --checkpoint -1 > logs/cutouts/100epochs_1e-4_mask_cutout_1_100x100_patch.log &

nohup python3.7 -u train.py --root "datasets/mask_cutout_1_100x100_patch_cut_mask" --device "cuda:1" --lr "1e-4" --epochs 100 --save "cutouts/100epochs_1e-4_mask_cutout_1_100x100_patch_cut_mask" --cutout --checkpoint -1 > logs/cutouts/100epochs_1e-4_mask_cutout_1_100x100_patch_cut_mask.log &

nohup python3.7 -u train.py --root "datasets/mask_cutout_1_100x100_patch_drop_initial" --device "cuda:2" --lr "1e-4" --epochs 100 --save "cutouts/100epochs_1e-4_mask_cutout_1_100x100_patch_drop_initial" --cutout --checkpoint -1 > logs/cutouts/100epochs_1e-4_mask_cutout_1_100x100_patch_drop_initial.log &

nohup python3.7 -u train.py --root "datasets/mask_cutout_1_100x100_patch_cut_mask_drop_initial" --device "cuda:2" --lr "1e-4" --epochs 100 --save "cutouts/100epochs_1e-4_mask_cutout_1_100x100_patch_cut_mask_drop_initial" --cutout --checkpoint -1 > logs/cutouts/100epochs_1e-4_mask_cutout_1_100x100_patch_cut_mask_drop_initial.log &
