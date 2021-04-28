python3.7 augment_supervised_dataset.py --root "datasets/mask_cutout_1_100x100_patch" --cnt 1 --min_size 100

python3.7 augment_supervised_dataset.py --root "datasets/mask_cutout_1_100x100_patch_cut_mask" --cnt 1 --min_size 100 --cut_mask

python3.7 augment_supervised_dataset.py --root "datasets/mask_cutout_1_100x100_patch_drop_initial" --cnt 1 --min_size 100 --drop_initial

python3.7 augment_supervised_dataset.py --root "datasets/mask_cutout_1_100x100_patch_cut_mask_drop_initial" --cnt 1 --min_size 100 --cut_mask --drop_initial

nohup python3.7 -u train.py --root "datasets/mask_cutout_1_100x100_patch" --device "cuda:1" --lr "1e-4" --epochs 50 --save "cutouts/50epochs_1e-4_mask_cutout_1_100x100_patch" --cutout --checkpoint -1 > logs/cutouts/50epochs_1e-4_mask_cutout_1_100x100_patch.log &

nohup python3.7 -u train.py --root "datasets/mask_cutout_1_100x100_patch_cut_mask" --device "cuda:1" --lr "1e-4" --epochs 50 --save "cutouts/50epochs_1e-4_mask_cutout_1_100x100_patch_cut_mask" --cutout --checkpoint -1 > logs/cutouts/50epochs_1e-4_mask_cutout_1_100x100_patch_cut_mask.log &

nohup python3.7 -u train.py --root "datasets/mask_cutout_1_100x100_patch_drop_initial" --device "cuda:2" --lr "1e-4" --epochs 50 --save "cutouts/50epochs_1e-4_mask_cutout_1_100x100_patch_drop_initial" --cutout --checkpoint -1 > logs/cutouts/50epochs_1e-4_mask_cutout_1_100x100_patch_drop_initial.log &

nohup python3.7 -u train.py --root "datasets/mask_cutout_1_100x100_patch_cut_mask_drop_initial" --device "cuda:2" --lr "1e-4" --epochs 50 --save "cutouts/50epochs_1e-4_mask_cutout_1_100x100_patch_cut_mask_drop_initial" --cutout --checkpoint -1 > logs/50epochs_1e-4_mask_cutout_1_100x100_patch_cut_mask_drop_initial.log &
