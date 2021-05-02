#segmentation with dropout

nohup python3.7 -u train_segmentation.py --dropout --device cuda:2 --lr 1e-4 --epochs 200 --save transfer/200epochs_1e-4_invert_dropout --checkpoint -1 --transfer autoencoder/1e-4_10epochs_invert.pt > logs/transfer/200epochs_1e-4_invert_dropout.log &

nohup python3.7 -u train_segmentation.py --dropout --no_skip --device cuda:2 --lr 1e-4 --epochs 200 --save transfer/200epochs_1e-4_invert_no_skip_dropout --checkpoint -1 --transfer autoencoder/1e-4_10epochs_invert_no_skip.pt > logs/transfer/200epochs_1e-4_invert_no_skip_dropout.log &

#segmentation zero skip dropout

nohup python3.7 -u train_segmentation.py --dropout --device cuda:2 --lr 1e-4 --epochs 200 --save transfer/200epochs_1e-4_invert_zero_skip_dropout --checkpoint -1 --transfer autoencoder/1e-4_10epochs_invert_zero_skip.pt > logs/transfer/200epochs_1e-4_invert_zero_skip_dropout.log &

nohup python3.7 -u train_segmentation.py --dropout --device cuda:2 --lr 1e-4 --epochs 200 --save transfer/200epochs_1e-4_zero_skip_dropout --checkpoint -1 --transfer autoencoder/1e-4_10epochs_zero_skip.pt > logs/transfer/200epochs_1e-4_zero_skip_dropout.log &

#reduce layers
#autoencoder

