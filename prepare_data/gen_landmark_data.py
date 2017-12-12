# coding: utf-8
import ipdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import random
import numpy.random as npr
from argparse import ArgumentParser
import pandas as pd

from BBox_utils import (
    getDataFromTxt,
    processImage,
    shuffle_in_unison_scary,
    BBox,
    IoU
)
from Landmark_utils import show_landmark, rotate, flip


def read_celeba_train_list(filepath, img_folder):
    """ Generate data from train list of celeba dataset.
        return [(img_path, bbox, landmark)]
            bbox: [left, top, right, bottom]
            landmark: [[x1, y1], [x2, y2], ...]
    """
    df = pd.read_csv(filepath)
    data = []
    for it, row in enumerate(df.itertuples()):

        if row.split == 2: # only use train (split=0) and val (split=1) images
            continue

        if np.any(np.isnan(np.array(row[2:]))):
            print('Skipping bad row {}: {}'.format(it, row))
            continue

        img_path = os.path.join(img_folder, row.image_id)
        bbox = [row.x_1, row.y_1, row.x_1 + row.width, row.y_1 + row.height] # x1, y1, x2, y2
        landmarks = np.array(row[6:16])
        landmarks = landmarks.reshape(5, 2) # [ [x1,y1], [x2,y2] ... ]

        if row.width <= 0 or row.height <= 0:
            print("Skipping bad bbox as row {}:\n {}.".format(it, row))
            continue

        #img = cv2.imread(img_path)
        #plt.imshow(img)
        #plt.scatter(landmarks[:, 0], landmarks[:, 1])
        #ax = plt.gca()
        #import matplotlib.patches as patches
        #ax.add_patch(patches.Rectangle(bbox[0:2], bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False))
        #plt.savefig('test-landmarks.png')

        data_cur = (img_path, BBox(bbox), landmarks)
        data.append(data_cur)

    return data


def read_lfwnet_train_list(filepath, img_folder):
    """ Generate data from train list of lfwnet dataset.
        return [(img_path, bbox, landmark)]
            bbox: [left, top, right, bottom]
            landmark: [[x1, y1], [x2, y2], ...]
    """

    with open(filepath, 'r') as fd:
        lines = fd.readlines()

    data = []
    for line in lines:
        components = line.strip().split(' ')
        img_path = os.path.join(img_folder, components[0])
        bbox = [int(components[i]) for i in [1, 3, 2, 4]] # x1, y1, x2, y2
        landmarks = np.array([float(v) for v in components[5:]])
        landmarks = landmarks.reshape(5, 2) # [ [x1,y1], [x2,y2] ... ]
        data_cur = (img_path, BBox(bbox), landmarks)
        data.append(data_cur)

    return data


def generate_data(data, save_folder, landmark_aug_save_folder, imsize, augment):

    savefile = os.path.join(save_folder, "landmark_{}_aug.txt".format(imsize))
    if os.path.exists(savefile):
        raise Exception("Save file already exists: {}".format(savefile))

    fid = open(savefile, 'w')
    index = 0

    print("Generating landmark data for {} images".format(len(data)))

    for img_idx, (imgPath, bbox, landmarkGt) in enumerate(data):

        if img_idx % 1000 == 0:
            print("Image {} / {}".format(img_idx, len(data)))

        F_imgs = []
        F_landmarks = []
        img = cv2.imread(imgPath)
        img_h, img_w, img_c = img.shape
        gt_box = np.array([bbox.left, bbox.top, bbox.right, bbox.bottom])
        f_face = img[bbox.top:bbox.bottom + 1, bbox.left:bbox.right + 1]
        f_face = cv2.resize(f_face, (imsize, imsize))

        # landmark: shifted and normalized by width and height
        landmark = (landmarkGt - gt_box[0:2]).astype(float) / np.array([[bbox.w, bbox.h]])

        F_imgs.append(f_face)
        F_landmarks.append(landmark.ravel())

        if augment:
            x1, y1, x2, y2 = gt_box
            gt_w = x2 - x1 + 1
            gt_h = y2 - y1 + 1

            if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                continue

            # random shift
            for i in range(10):
                bbox_size = npr.randint(
                    int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                delta_x = npr.randint(-gt_w * 0.2, gt_w * 0.2)
                delta_y = npr.randint(-gt_h * 0.2, gt_h * 0.2)
                nx1 = max(x1 + gt_w / 2 - bbox_size / 2 + delta_x, 0)
                ny1 = max(y1 + gt_h / 2 - bbox_size / 2 + delta_y, 0)

                nx2 = nx1 + bbox_size
                ny2 = ny1 + bbox_size
                if nx2 > img_w or ny2 > img_h:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])
                cropped_im = img[ny1:ny2 + 1, nx1:nx2 + 1, :]
                resized_im = cv2.resize(cropped_im, (imsize, imsize))

                iou = IoU(crop_box, np.expand_dims(gt_box, 0))
                if iou > 0.65:
                    F_imgs.append(resized_im)
                    # normalize
                    landmark = (landmarkGt - np.array([[nx1, ny1]])) / bbox_size
                    F_landmarks.append(landmark.reshape(10))
                    landmark_ = F_landmarks[-1].reshape(-1, 2)
                    bbox = BBox([nx1, ny1, nx2, ny2])

                    # mirror
                    if random.choice([0, 1]) > 0:
                        face_flipped, landmark_flipped = flip(
                            resized_im, landmark_)
                        face_flipped = cv2.resize(face_flipped, (imsize, imsize))
                        # c*h*w
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
                    # rotate
                    if random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(
                            img, bbox, bbox.reprojectLandmark(landmark_), 5)  # 逆时针旋转
                        # landmark_offset
                        landmark_rotated = bbox.projectLandmark(
                            landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(
                            face_rotated_by_alpha, (imsize, imsize))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(10))

                        # flip
                        face_flipped, landmark_flipped = flip(
                            face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (imsize, imsize))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))

                    # inverse clockwise rotation
                    if random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(
                            img, bbox, bbox.reprojectLandmark(landmark_), -5)  # 顺时针旋转
                        landmark_rotated = bbox.projectLandmark(
                            landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(
                            face_rotated_by_alpha, (imsize, imsize))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(10))

                        face_flipped, landmark_flipped = flip(
                            face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (imsize, imsize))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))


        for i in range(len(F_imgs)):

            if np.any(F_landmarks[i] <= 0) or np.any(F_landmarks[i] >= 1):
                continue

            # save image
            img_filepath = os.path.join(landmark_aug_save_folder, "{}.jpg".format(index))
            cv2.imwrite(img_filepath, F_imgs[i])

            # save meta information
            landmark_str = F_landmarks[i].tolist()
            landmark_str = map(str, landmark_str)
            landmark_str = " ".join(landmark_str)
            line = img_filepath + " -2 " + landmark_str + "\n"
            fid.write(line)

            index = index + 1

    fid.close()
    return F_imgs, F_landmarks


def process_dataset(dataset_type, net_type):

    if net_type == 'PNet':
        imsize = 12
    elif net_type == 'RNet':
        imsize = 24
    elif net_type == 'ONet':
        imsize = 48
    else:
        raise Exception("Invalid argument: {}".format(net_type))

    save_folder = str(imsize)

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    landmark_aug_save_folder = os.path.join(
        save_folder,
        "train_{}_landmark_aug".format(net_type))

    if not os.path.exists(landmark_aug_save_folder):
        os.mkdir(landmark_aug_save_folder)

    if dataset_type == 'lfwnet':
        data = read_lfwnet_train_list('lfw_net_image_list.txt', '.')
        augment = True
    elif dataset_type == 'celeba':
        data = read_celeba_train_list('celeba_image_list.txt', '.')
        augment = False
    else:
        raise Exception("Invalid argument: {}".format(dataset_type))

    imgs, landmarks = generate_data(
        data,
        save_folder,
        landmark_aug_save_folder,
        imsize,
        augment)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset_type', help="lfwnet or celeba")
    parser.add_argument('net_type', help="PNet, RNet or ONet")
    args = parser.parse_args()

    process_dataset(args.dataset_type, args.net_type)

