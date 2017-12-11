# encoding: utf-8

"""
@version: 1.0
@author: liuhengli
@license: Apache Licence
@software: PyCharm
@file: train.py
@time: 2017/7/25 9:04
"""

import os
import numpy as np
import cv2


def read_annotation(base_dir, label_path):
    """
    read label file
    :param dir: path
    :return:
    """
    data = dict()
    images = []
    bboxes = []
    labelfile = open(label_path, 'r')
    while True:
        # image path
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = base_dir + '/WIDER_train/images/' + imagepath
        images.append(imagepath)
        # face numbers
        nums = labelfile.readline().strip('\n')
        # im = cv2.imread(imagepath)
        # h, w, c = im.shape
        one_image_bboxes = []
        for i in range(int(nums)):
            # text = ''
            # text = text + imagepath
            bb_info = labelfile.readline().strip('\n').split(' ')
            # only need x, y, w, h
            face_box = [float(bb_info[i]) for i in range(4)]
            # text = text + ' ' + str(face_box[0] / w) + ' ' + str(face_box[1] / h)
            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2]
            ymax = ymin + face_box[3]
            # text = text + ' ' + str(xmax / w) + ' ' + str(ymax / h)
            one_image_bboxes.append([xmin, ymin, xmax, ymax])
            # f.write(text + '\n')
        bboxes.append(one_image_bboxes)


    data['images'] = images#all image pathes
    data['bboxes'] = bboxes#all image bboxes
    # f.close()
    return data

def read_and_write_annotation(base_dir, dir):
    """
    read label file
    :param dir: path
    :return:
    """
    data = dict()
    images = []
    bboxes = []
    labelfile = open(dir, 'r')
    f = open('/home/thinkjoy/data/mtcnn_data/imagelists/train.txt', 'w')
    while True:
        # image path
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = base_dir + '/WIDER_train/images/' + imagepath
        images.append(imagepath)
        # face numbers
        nums = labelfile.readline().strip('\n')
        im = cv2.imread(imagepath)
        h, w, c = im.shape
        one_image_bboxes = []
        for i in range(int(nums)):
            text = ''
            text = text + imagepath
            bb_info = labelfile.readline().strip('\n').split(' ')
            # only need x, y, w, h
            face_box = [float(bb_info[i]) for i in range(4)]
            text = text + ' ' + str(face_box[0] / w) + ' ' + str(face_box[1] / h)
            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2] - 1
            ymax = ymin + face_box[3] - 1
            text = text + ' ' + str(xmax / w) + ' ' + str(ymax / h)
            one_image_bboxes.append([xmin, ymin, xmax, ymax])
            f.write(text + '\n')
        bboxes.append(one_image_bboxes)


    data['images'] = images
    data['bboxes'] = bboxes
    f.close()
    return data

def get_path(base_dir, filename):
    return os.path.join(base_dir, filename)


if __name__ == '__main__':
    dir = '/media/thinkjoy/新加卷/dataset/widerface/wider_face_split/wider_face_train_bbx_gt.txt'
    base_dir = '/media/thinkjoy/新加卷/dataset/widerface'
    data = read_annotation(base_dir, dir)
    print('\n')
    print(data['images'])
    print("============")
    print('\n')
    print(data['bboxes'])





