python3.7 augment_supervised_dataset.py --root "datasets/mask_cutout_1_100x100_patch" --cnt 1 --min_size 100

python3.7 augment_supervised_dataset.py --root "datasets/mask_cutout_1_100x100_patch_cut_mask" --cnt 1 --min_size 100 --cut_mask

python3.7 augment_supervised_dataset.py --root "datasets/mask_cutout_1_100x100_patch_drop_initial" --cnt 1 --min_size 100 --drop_initial

python3.7 augment_supervised_dataset.py --root "datasets/mask_cutout_1_100x100_patch_cut_mask_drop_initial" --cnt 1 --min_size 100 --cut_mask --drop_initial