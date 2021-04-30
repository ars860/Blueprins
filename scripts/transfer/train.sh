nohup python3.7 -u train.py --device cuda:2 --lr 1e-4 --epochs 200 --save transfer/200epochs_1e-4_gauss_05 --checkpoint -1 --transfer autoencoder/1e-4_gauss_05_10epochs.pt > logs/transfer/200epochs_1e-4_gauss_05.log &

nohup python3.7 -u train.py --device cuda:2 --lr 1e-4 --epochs 200 --save transfer/200epochs_1e-4 --checkpoint -1 --transfer autoencoder/1e-4_10epochs.pt > logs/transfer/200epochs_1e-4.log &
