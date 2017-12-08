# coding:utf-8
from mtcnn_model import P_Net, R_Net, O_Net
from train import train

if __name__ == '__main__':
    net = 'ONet'
    dataset_folder = '../prepare_data/imglists/{}'.format(net)
    experiment_folder = '../data/MTCNN_model/{}_landmark'.format(net)

    end_epoch = 22
    display = 10
    lr = 0.01
    train(
        O_Net,
        net,
        experiment_folder,
        dataset_folder,
        end_epoch,
        display=display,
        base_lr=lr)
