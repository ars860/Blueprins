python3.7 augment_supervised_dataset.py --root "datasets/mask_cutout_100_10x10_gray_patch" --cnt 100 --min_size 10 --max_size 10 --val 0.5;

python3.7 augment_supervised_dataset.py --root "datasets/mask_cutout_100_10x10_gray_patch_cut_mask" --cnt 100 --min_size 10 --max_size 10 --val 0.5 --cut_mask;

python3.7 augment_supervised_dataset.py --root "datasets/mask_cutout_100_10x10_gray_patch_drop_initial" --cnt 100 --min_size 10 --max_size 10 --val 0.5 --drop_initial;

python3.7 augment_supervised_dataset.py --root "datasets/mask_cutout_100_10x10_gray_patch_cut_mask_drop_initial" --cnt 100 --min_size 10 --max_size 10 --val 0.5 --cut_mask --drop_initial;