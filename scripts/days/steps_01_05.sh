#autoencoders

nohup python3.7 -u autoencoder_train.py --invert --device cuda:2 --lr 1e-4 --epochs 10 --save autoencoder/1e-4_10epochs_invert > logs/autoencoder/1e-4_10epochs_invert.log &

nohup python3.7 -u autoencoder_train.py --invert --no_skip --device cuda:2 --lr 1e-4 --epochs 10 --save autoencoder/1e-4_10epochs_invert_no_skip > logs/autoencoder/1e-4_10epochs_invert_no_skip.log &

nohup python3.7 -u autoencoder_train.py --no_skip --device cuda:2 --lr 1e-4 --epochs 10 --save autoencoder/1e-4_10epochs_no_skip > logs/autoencoder/1e-4_10epochs_no_skip.log &

#segmentation

nohup python3.7 -u train_segmentation.py --device cuda:2 --lr 1e-4 --epochs 200 --save transfer/200epochs_1e-4_invert --checkpoint -1 --transfer autoencoder/1e-4_10epochs_invert.pt > logs/transfer/200epochs_1e-4_invert.log &

nohup python3.7 -u train_segmentation.py --no_skip --device cuda:2 --lr 1e-4 --epochs 200 --save transfer/200epochs_1e-4_invert_no_skip --checkpoint -1 --transfer autoencoder/1e-4_10epochs_invert_no_skip.pt > logs/transfer/200epochs_1e-4_invert_no_skip.log &

nohup python3.7 -u train_segmentation.py --no_skip --device cuda:2 --lr 1e-4 --epochs 200 --save transfer/200epochs_1e-4_no_skip --checkpoint -1 --transfer autoencoder/1e-4_10epochs_no_skip.pt > logs/transfer/200epochs_1e-4_no_skip.log &

#new augmentations

python3.7 augment_supervised_dataset.py --root datasets/200_5-15x5-15_x4_drop_initial_vh_shuffle_123 --cnt 200 --min_size 5 --max_size 15 --drop_initial --times 4 --v --h --shuffle 123

#fixed autoencoder invert test loss

nohup python3.7 -u autoencoder_train.py --invert --device cuda:2 --lr 1e-4 --epochs 10 --save autoencoder/1e-4_10epochs_invert_correct_test > logs/autoencoder/1e-4_10epochs_invert_correct_test.log &

nohup python3.7 -u autoencoder_train.py --invert --no_skip --device cuda:2 --lr 1e-4 --epochs 10 --save autoencoder/1e-4_10epochs_invert_no_skip_correct_test > logs/autoencoder/1e-4_10epochs_invert_no_skip_correct_test.log &

#segmentaion on augmented dataset

nohup python3.7 -u train_segmentation.py --root datasets/200_5-15x5-15_x4_drop_initial_vh_shuffle_123 --device cuda:2 --lr 1e-4 --epochs 200 --save transfer/200_5-15x5-15_x4_drop_initial_vh_shuffle_123/200epochs_1e-4_invert --checkpoint -1 --transfer autoencoder/1e-4_10epochs_invert.pt > logs/transfer/200_5-15x5-15_x4_drop_initial_vh_shuffle_123/200epochs_1e-4_invert.log &

nohup python3.7 -u train_segmentation.py --root datasets/200_5-15x5-15_x4_drop_initial_vh_shuffle_123 --no_skip --device cuda:2 --lr 1e-4 --epochs 200 --save transfer/200_5-15x5-15_x4_drop_initial_vh_shuffle_123/200epochs_1e-4_invert_no_skip --checkpoint -1 --transfer autoencoder/1e-4_10epochs_invert_no_skip.pt > logs/transfer/200_5-15x5-15_x4_drop_initial_vh_shuffle_123/200epochs_1e-4_invert_no_skip.log &

nohup python3.7 -u train_segmentation.py --root datasets/200_5-15x5-15_x4_drop_initial_vh_shuffle_123 --no_skip --device cuda:2 --lr 1e-4 --epochs 200 --save transfer/200_5-15x5-15_x4_drop_initial_vh_shuffle_123/200epochs_1e-4_no_skip --checkpoint -1 --transfer autoencoder/1e-4_10epochs_no_skip.pt > logs/transfer/200_5-15x5-15_x4_drop_initial_vh_shuffle_123/200epochs_1e-4_no_skip.log &

#autoencoders zero_skip

nohup python3.7 -u autoencoder_train.py --invert --zero_skip --device cuda:1 --lr 1e-4 --epochs 10 --save autoencoder/1e-4_10epochs_invert_zero_skip > logs/autoencoder/1e-4_10epochs_invert_zero_skip.log &

nohup python3.7 -u autoencoder_train.py --zero_skip --device cuda:1 --lr 1e-4 --epochs 10 --save autoencoder/1e-4_10epochs_zero_skip > logs/autoencoder/1e-4_10epochs_zero_skip.log &

#segmentaion zero_skip

nohup python3.7 -u train_segmentation.py --device cuda:2 --lr 1e-4 --epochs 200 --save transfer/200epochs_1e-4_invert_zero_skip --checkpoint -1 --transfer autoencoder/1e-4_10epochs_invert_zero_skip.pt > logs/transfer/200epochs_1e-4_invert_zero_skip.log &

nohup python3.7 -u train_segmentation.py --device cuda:2 --lr 1e-4 --epochs 200 --save transfer/200epochs_1e-4_zero_skip --checkpoint -1 --transfer autoencoder/1e-4_10epochs_zero_skip.pt > logs/transfer/200epochs_1e-4_zero_skip.log &

#no transfer on augmented

nohup python3.7 -u train_segmentation.py --root datasets/200_5-15x5-15_x4_drop_initial_vh_shuffle_123 --device cuda:2 --lr 1e-4 --epochs 200 --save no_transfer/200_5-15x5-15_x4_drop_initial_vh_shuffle_123/200epochs_1e-4 --checkpoint -1 > logs/no_transfer/200_5-15x5-15_x4_drop_initial_vh_shuffle_123/200epochs_1e-4.log &
