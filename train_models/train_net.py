# coding:utf-8
from mtcnn_model import P_Net, R_Net, O_Net
from train import train
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('net', help='PNet, RNet, ONet')
    args = parser.parse_args()

    net = args.net

    if net == 'PNet':
        end_epoch = 30
        net_factory = P_Net
    elif net == 'RNet':
        end_epoch = 22
        net_factory = R_Net
    elif net == 'ONet':
        end_epoch = 22
        net_factory = O_Net
    else:
        raise Exception("Invalid argument: {}".format(net))

    lr = 0.01
    display = 1000
    dataset_folder = '../prepare_data/imglists/{}'.format(net)
    experiment_folder = '../data/MTCNN_model/{}_landmark'.format(net)

    train(
        net_factory,
        net,
        experiment_folder,
        dataset_folder,
        end_epoch,
        display=display,
        base_lr=lr)
