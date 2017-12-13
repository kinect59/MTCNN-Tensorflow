# coding:utf-8
import mtcnn_model
import mtcnn_collaborative_model
#from mtcnn_model import P_Net, R_Net, O_Net
from train import train
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('net', help='PNet, RNet, ONet')
    parser.add_argument('--collaborative', action='store_true',
                        help='Use collaboative MTL')
    args = parser.parse_args()
    net = args.net

    if args.collaborative:
        PNet_factory = mtcnn_collaborative_model.P_Net
        RNet_factory = mtcnn_collaborative_model.R_Net
        ONet_factory = mtcnn_collaborative_model.O_Net
        experiment_folder = '../data/MTCNN_collaborative_model/{}_landmark'.format(net)
    else:
        PNet_factory = mtcnn_model.P_Net
        RNet_factory = mtcnn_model.R_Net
        ONet_factory = mtcnn_model.O_Net
        experiment_folder = '../data/MTCNN_model/{}_landmark'.format(net)

    if net == 'PNet':
        end_epoch = 30
        net_factory = PNet_factory
    elif net == 'RNet':
        end_epoch = 22
        net_factory = RNet_factory
    elif net == 'ONet':
        end_epoch = 22
        net_factory = ONet_factory
    else:
        raise Exception("Invalid argument: {}".format(net))

    lr = 0.01
    display = 1000
    dataset_folder = '../prepare_data/imglists/{}'.format(net)

    train(
        net_factory,
        net,
        experiment_folder,
        dataset_folder,
        end_epoch,
        display=display,
        base_lr=lr)
