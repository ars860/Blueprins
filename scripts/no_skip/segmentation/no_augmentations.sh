nohup python3.7 -u train.py --no_skip --device cuda:2 --lr 1e-4 --epochs 200 --save no_skip/segmentation/200epochs_1e-4_sigmoid --checkpoint -1 --transfer no_skip/autoencoder/1e-4_10epochs_sigmoid.pt > logs/no_skip/segmentation/200epochs_1e-4_sigmoid.log &

nohup python3.7 -u train.py --no_skip --device cuda:2 --lr 1e-4 --epochs 200 --save no_skip/segmentation/200epochs_1e-4_gauss_05 --checkpoint -1 --transfer no_skip/autoencoder/1e-4_gauss_05_10epochs.pt > logs/no_skip/segmentation/200epochs_1e-4_gauss_05.log &
