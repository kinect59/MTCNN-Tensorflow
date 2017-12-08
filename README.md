## Description
This work is used for reproduce MTCNN, a Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks.

## Prerequisites
1. You need CUDA-compatible GPUs to train the model.

## Dependencies
* Tensorflow 1.2.1
* Python 2.7
* Ubuntu 16.04
* Cuda 8.0

## Prepare For Training Data
1. Download the [WIDER Face Dataset](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/). Unzip it and put the `WIDER_train` folder in the `prepare_data` folder.
2. Download the [Landmark Dataset](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm). Unzip it and put both the `lfw_5590` and `net_7876` into the `prepare_data` folder.
3. Run `gen_12net_data.py` to generate training data(Face Detection Part) for **PNet**.
4. Run `gen_landmark_data.py PNet` to generate training data(Face Landmark Detection Part) for **PNet**.
5. Run `gen_imglist.py PNet` to merge two parts of training data.
6. Run `gen_tfrecords.py PNet` to generate tfrecord for **PNet**.
7. In `train_models/`, run `train_net.py PNet` to train **PNet**.
8. After training **PNet**, run `gen_hard_example.py --test_mode PNet` to generate training data(Face Detection Part) for **RNet**.
9. Run `gen_landmark_data.py RNet` to generate training data(Face Landmark Detection Part) for **RNet**.
10. Run `gen_imglist.py PNet` to merge two parts of training data.
11. Run `gen_tfrecords.py RNet` to generate tfrecords for **RNet**.
12. In `train_models/`, run `train_net.py RNet` to train **RNet**.
13. After training **RNet**, run `gen_hard_example.py --test_mode RNet` to generate training data(Face Detection Part) for **ONet**.
14. Run `gen_landmark_data.py ONet` to generate training data(Face Landmark Detection Part) for **ONet**.
15. Run `gen_imglist.py ONet` to merge two parts of training data.
16. Run `gen_tfrecords.py ONet` to generate tfrecords for **ONet**.
17. In `train_models/`, run `train_net.py ONet` to train **ONet**.

## Some Details
* When training **PNet**,I merge four parts of data(pos,part,landmark,neg) into one tfrecord,since their total number radio is almost 1:1:1:3.But when training **RNet** and **ONet**,I generate four tfrecords,since their total number is not balanced.During training,I read 64 samples from pos,part and landmark tfrecord and read 192 samples from neg tfrecord to construct mini-batch.
* It's important for **PNet** and **RNet** to keep high recall radio.When using well-trained **PNet** to generate training data for **RNet**,I can get 14w+ pos samples.When using well-trained **RNet** to generate training data for **ONet**,I can get 19w+ pos samples.
* Since **MTCNN** is a Multi-task Network,we should pay attention to the format of training data.The format is:
 
  [path to image][cls_label][bbox_label][landmark_label]
  
  For pos sample,cls_label=1,bbox_label(calculate),landmark_label=[0,0,0,0,0,0,0,0,0,0].

  For part sample,cls_label=-1,bbox_label(calculate),landmark_label=[0,0,0,0,0,0,0,0,0,0].
  
  For landmark sample,cls_label=-2,bbox_label=[0,0,0,0],landmark_label(calculate).  
  
  For neg sample,cls_label=0,bbox_label=[0,0,0,0],landmark_label=[0,0,0,0,0,0,0,0,0,0].  

* Since the training data for landmark is less.I use transform,random rotate and random flip to conduct data augment(the result of landmark detection is not that good).

## Result

![result1.png](https://i.loli.net/2017/08/30/59a6b65b3f5e1.png)

![result2.png](https://i.loli.net/2017/08/30/59a6b6b4efcb1.png)

![result3.png](https://i.loli.net/2017/08/30/59a6b6f7c144d.png)

![reult4.png](https://i.loli.net/2017/08/30/59a6b72b38b09.png)

![result5.png](https://i.loli.net/2017/08/30/59a6b76445344.png)

![result6.png](https://i.loli.net/2017/08/30/59a6b79d5b9c7.png)

![result7.png](https://i.loli.net/2017/08/30/59a6b7d82b97c.png)

![result8.png](https://i.loli.net/2017/08/30/59a6b7ffad3e2.png)

![result9.png](https://i.loli.net/2017/08/30/59a6b843db715.png)

**Result on FDDB**
![result10.png](https://i.loli.net/2017/08/30/59a6b875f1792.png)

## License
MIT LICENSE

## References
1. Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, Yu Qiao , " Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks," IEEE Signal Processing Letter
2. [MTCNN-MXNET](https://github.com/Seanlinx/mtcnn)
3. [MTCNN-CAFFE](https://github.com/CongWeilin/mtcnn-caffe)
4. [deep-landmark](https://github.com/luoyetx/deep-landmark)
