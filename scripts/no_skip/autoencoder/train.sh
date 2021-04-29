nohup python3.7 -u autoencoder_train.py --no_skip --device cuda:2 --lr 1e-4 --epochs 10 --save no_skip/autoencoder/1e-4_10epochs > logs/no_skip/autoencoder/1e-4_10epochs.log &

nohup python3.7 -u autoencoder_train.py --no_skip --device cuda:2 --lr 1e-4 --epochs 10 --gaussian_noise 0.5 --save no_skip/autoencoder/1e-4_gauss_05_10epochs > logs/no_skip/autoencoder/1e-4_gauss_05_10epochs.log &
