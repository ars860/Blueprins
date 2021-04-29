python3.7 augment_supervised_dataset.py --root "datasets/mask_cutout_100_10x10_gray_patch_cut_mask_drop_initial" --cnt 100 --min_size 10 --max_size 10 --val 0.5 --cut_mask --drop_initial;

python3.7 augment_supervised_dataset.py --root "datasets/mask_cutout_1_100x100_patch_cut_mask_drop_initial" --cnt 1 --min_size 100 --cut_mask --drop_initial;

python3.7 augment_supervised_dataset.py --root "datasets/mask_cutout_1_100x100_gray_patch_cut_mask_drop_initial" --cnt 1 --min_size 100 --cut_mask --drop_initial --val 0.5;

python3.7 augment_supervised_dataset.py --root "datasets/mask_cutout_100_10x10_patch_cut_mask_drop_initial" --cnt 100 --min_size 10 --max_size 10 --cut_mask --drop_initial;

python3.7 augment_supervised_dataset.py --root "datasets/mask_cutout_100_10x10_gray_patch_x5_cut_mask_drop_initial" --times 5 --cnt 100 --min_size 10 --max_size 10 --val 0.5 --cut_mask --drop_initial;

python3.7 augment_supervised_dataset.py --root "datasets/mask_cutout_100_10x10_black_patch_x5_cut_mask_drop_initial" --times 5 --cnt 100 --min_size 10 --max_size 10 --val 0.0 --cut_mask --drop_initial;
