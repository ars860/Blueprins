nohup python3.7 -u autoencoder_train.py --device cuda:2 --lr 1e-4 --epochs 10 --save autoencoder/1e-4_10epochs > logs/autoencoder/1e-4_10epochs.log &

nohup python3.7 -u autoencoder_train.py --device cuda:2 --lr 1e-4 --epochs 10 --gaussian_noise 0.5 --save autoencoder/1e-4_gauss_05_10epochs > logs/autoencoder/1e-4_gauss_05_10epochs.log &
