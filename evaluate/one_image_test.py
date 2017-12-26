# coding:utf-8
import sys
sys.path.append('..')
import pandas as pd
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models import mtcnn_model
from train_models import mtcnn_collaborative_model
from prepare_data.loader import TestLoader
from prepare_data.Landmark_utils import show_landmarks, lfs
from prepare_data.BBox_utils import show_bbox, IoU


collaborative = True
if collaborative:
    PNet_factory = mtcnn_collaborative_model.P_Net
    RNet_factory = mtcnn_collaborative_model.R_Net
    ONet_factory = mtcnn_collaborative_model.O_Net
    #ONet_factory = mtcnn_model.O_Net
    prefix = [
        '../data/MTCNN_collaborative_model_mult_wd_1e-4/PNet_landmark/PNet',
        '../data/MTCNN_collaborative_model_mult_wd_1e-4/RNet_landmark/RNet',
        '../data/MTCNN_collaborative_model/ONet_landmark/ONet'
    #    '../data/MTCNN_collaborative_model/ONet_landmark_wd_1e-4/ONet'
    #    '../data/MTCNN_model_mult_wd_1e-4/ONet_landmark/ONet'
    ]
else:
    PNet_factory = mtcnn_model.P_Net
    RNet_factory = mtcnn_model.R_Net
    ONet_factory = mtcnn_model.O_Net
    prefix = [
        '../data/MTCNN_model_mult_wd_1e-4/PNet_landmark/PNet',
        '../data/MTCNN_model_mult_wd_1e-4/RNet_landmark/RNet',
        '../data/MTCNN_model_mult_wd_1e-4/ONet_landmark/ONet'
    ]
    #prefix = [
    #    '../data/MTCNN_model_lfwnet/PNet_landmark/PNet',
    #    '../data/MTCNN_model_lfwnet/RNet_landmark/RNet',
    #    '../data/MTCNN_model_lfwnet/ONet_landmark/ONet']


test_mode = "ONet"
thresh = [0.9, 0.6, 0.7]
#thresh = [0.5, 0.5, 0.5]
min_face_size = 24
stride = 2
slide_window = False
shuffle = False
detectors = [None, None, None]
show_predictions = False

#epoch = [30, 22, 22]
epoch = [70, 70, 70]
batch_size = [2048, 256, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]


# load pnet model
if slide_window:
    PNet = Detector(PNet_factory, 12, batch_size[0], model_path[0])
else:
    PNet = FcnDetector(PNet_factory, model_path[0])
detectors[0] = PNet

# load rnet model
if test_mode in ["RNet", "ONet"]:
    RNet = Detector(RNet_factory, 24, batch_size[1], model_path[1])
    detectors[1] = RNet

# load onet model
if test_mode == "ONet":
    ONet = Detector(ONet_factory, 48, batch_size[2], model_path[2])
    detectors[2] = ONet

mtcnn_detector = MtcnnDetector(
    detectors=detectors,
    min_face_size=min_face_size,
    stride=stride,
    threshold=thresh,
    slide_window=slide_window)

#aflw_df = pd.read_csv('../prepare_data/aflw_image_list.txt')
#img_list = [os.path.join('..', 'prepare_data', p) for p in list(aflw_df.image_id)]
#img_list = img_list[10:20]
#test_data = TestLoader(img_list)


celeba_df = pd.read_csv('../prepare_data/celeba_image_list.txt')
celeba_df = celeba_df[celeba_df.split == 2]
celeba_df = celeba_df[0:50]
img_list = [os.path.join('..', 'prepare_data', p) for p in list(celeba_df.image_id)]
celeba_df.image_id = img_list
celeba_df = celeba_df.reset_index()

test_data = TestLoader(img_list)
all_boxes, all_landmarks = mtcnn_detector.detect_face(test_data)

N = len(celeba_df)
all_mean_dists = np.zeros(N)
all_ratios = np.zeros(N)

for it, row in celeba_df.iterrows():
    image_id = row.image_id
    gt_bbox = np.array(list(row[2:6]))
    gt_landmarks = np.array(list(row[6:16]))

    if all_landmarks[it].size == 0:
        ratio = 1
        dist = np.nan
    else:
        mean_dists, ratios = lfs(all_landmarks[it], gt_landmarks)
        ratio = ratios.min()
        dist = mean_dists.min()

    all_mean_dists[it] = dist
    all_ratios[it] = ratio
    print("Image {}: dist = {}, ratio = {}".format(it, dist, ratio))

    if show_predictions:
        image = plt.imread(image_id)
        show_bbox(gt_bbox, 'blue')
        for bbox in all_boxes[it]:
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]
            show_bbox(bbox)

        for landmarks in all_landmarks[it]:
            show_landmarks(landmarks)

        #cv2.imwrite("result_landmark/%d.png" %(it),image)
        plt.axis('off')
        plt.imshow(image)
        plt.show()

all_failures = all_ratios > 0.1

print('Mean of mean dists: {}'.format(np.nanmean(all_mean_dists)))
print('Mean of failures: {}'.format(np.mean(all_failures)))
