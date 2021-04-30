python3.7 -u augment_supervised_dataset.py --root datasets/cutout_200_10x10_x4_drop_initial  --cnt 200 --min_size 10 --max_size 10 --drop_initial --times 4

nohup python3.7 -u train.py --cutout --root datasets/cutout_200_10x10_x4_drop_initial --device cuda:2 --lr 1e-4 --epochs 200 --save transfer/200epochs_1e-4_cutout_200_10x10_x4_drop_initial --checkpoint -1 --transfer autoencoder/1e-4_10epochs.pt > logs/transfer/200epochs_1e-4_cutout_200_10x10_x4_drop_initial.log &

python3.7 -u augment_supervised_dataset.py --root datasets/cutout_1_50x50_x4_drop_initial  --cnt 1 --min_size 50 --max_size 50 --drop_initial --times 4

nohup python3.7 -u train.py --cutout --root datasets/cutout_1_50x50_x4_drop_initial --device cuda:2 --lr 1e-4 --epochs 200 --save transfer/200epochs_1e-4_cutout_1_50x50_x4_drop_initial --checkpoint -1 --transfer autoencoder/1e-4_10epochs.pt > logs/transfer/200epochs_1e-4_cutout_1_50x50_x4_drop_initial.log &

python3.7 -u augment_supervised_dataset.py --root datasets/cutout_1_50x50_x4_drop_initial_cut_mask  --cnt 1 --min_size 50 --max_size 50 --drop_initial --cut_mask --times 4

nohup python3.7 -u train.py --cutout --root datasets/cutout_1_50x50_x4_drop_initial_cut_mask --device cuda:2 --lr 1e-4 --epochs 200 --save transfer/200epochs_1e-4_cutout_1_50x50_x4_drop_initial_cut_mask --checkpoint -1 --transfer autoencoder/1e-4_10epochs.pt > logs/transfer/200epochs_1e-4_cutout_1_50x50_x4_drop_initial_cut_mask.log &
