# coding:utf-8
import ipdb
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
num_keep_radio = 0.7


def prelu(inputs):
    alphas = tf.get_variable("alphas",
                             shape=inputs.get_shape()[-1],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5
    return pos + neg


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def cls_ohem(cls_prob, label):
    zeros = tf.zeros_like(label)
    # label=-1 --> label=0net_factory
    label_filter_invalid = tf.where(tf.less(label, 0), zeros, label)
    num_cls_prob = tf.size(cls_prob)
    cls_prob_reshape = tf.reshape(cls_prob, [num_cls_prob, -1])
    label_int = tf.cast(label_filter_invalid, tf.int32)
    num_row = tf.to_int32(cls_prob.get_shape()[0])
    row = tf.range(num_row) * 2
    indices_ = row + label_int
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.log(label_prob + 1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob, dtype=tf.float32)
    valid_inds = tf.where(label < zeros, zeros, ones)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid * num_keep_radio, dtype=tf.int32)
    # set 0 to invalid sample
    loss = loss * valid_inds
    loss, _ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)


def bbox_ohem_smooth_L1_loss(bbox_pred, bbox_target, label):
    sigma = tf.constant(1.0)
    threshold = 1.0 / (sigma**2)
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    valid_inds = tf.where(
        label != zeros_index,
        tf.ones_like(
            label,
            dtype=tf.float32),
        zeros_index)
    abs_error = tf.abs(bbox_pred - bbox_target)
    loss_smaller = 0.5 * ((abs_error * sigma)**2)
    loss_larger = abs_error - 0.5 / (sigma**2)
    smooth_loss = tf.reduce_sum(
        tf.where(
            abs_error < threshold,
            loss_smaller,
            loss_larger),
        axis=1)
    keep_num = tf.cast(
        tf.reduce_sum(valid_inds) *
        num_keep_radio,
        dtype=tf.int32)
    smooth_loss = smooth_loss * valid_inds
    _, k_index = tf.nn.top_k(smooth_loss, k=keep_num)
    smooth_loss_picked = tf.gather(smooth_loss, k_index)
    return tf.reduce_mean(smooth_loss_picked)


def bbox_ohem_orginal(bbox_pred, bbox_target, label):
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    # pay attention :there is a bug!!!!
    valid_inds = tf.where(
        label != zeros_index,
        tf.ones_like(
            label,
            dtype=tf.float32),
        zeros_index)
    #(batch,)
    square_error = tf.reduce_sum(tf.square(bbox_pred - bbox_target), axis=1)
    # keep_num scalar
    keep_num = tf.cast(
        tf.reduce_sum(valid_inds) *
        num_keep_radio,
        dtype=tf.int32)
    # keep valid index square_error
    square_error = square_error * valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)
# label=1 or label=-1 then do regression


def bbox_ohem(bbox_pred, bbox_target, label):
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label, dtype=tf.float32)
    valid_inds = tf.where(tf.equal(tf.abs(label), 1), ones_index, zeros_index)
    #(batch,)
    square_error = tf.square(bbox_pred - bbox_target)
    square_error = tf.reduce_sum(square_error, axis=1)
    # keep_num scalar
    num_valid = tf.reduce_sum(valid_inds)
    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    # keep valid index square_error
    square_error = square_error * valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)


def landmark_ohem(landmark_pred, landmark_target, label):
    # keep label =-2  then do landmark detection
    ones = tf.ones_like(label, dtype=tf.float32)
    zeros = tf.zeros_like(label, dtype=tf.float32)
    valid_inds = tf.where(tf.equal(label, -2), ones, zeros)
    square_error = tf.square(landmark_pred - landmark_target)
    square_error = tf.reduce_sum(square_error, axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error * valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)


def cal_accuracy(cls_prob, label):
    pred = tf.argmax(cls_prob, axis=1)
    label_int = tf.cast(label, tf.int64)
    cond = tf.where(tf.greater_equal(label_int, 0))
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_int, picked)
    pred_picked = tf.gather(pred, picked)
    accuracy_op = tf.reduce_mean(
        tf.cast(
            tf.equal(
                label_picked,
                pred_picked),
            tf.float32))
    return accuracy_op


def collaborative_block(inputs, nf, training, scope):

    def conv2d(x, nf, fs):
        y = slim.conv2d(x, nf, [fs, fs], 1, 'SAME', activation_fn=None,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=None,
                        weights_regularizer=slim.l2_regularizer(0.0005))
        return y

    def bn(x, training):
        y = slim.batch_norm(x, 0.999, True, True, 1e-5, is_training=training)
        return y

    def aggregation(x, n_out, training):
        with tf.variable_scope('aggregation'):
            z = x

            # version 1
            #z = conv2d(z, n_out, 1)
            #z = bn(z, training)
            #z = tf.nn.relu(z)
            #z = conv2d(z, n_out, 3)
            #z = bn(z, training)

            # version 2
            z = conv2d(z, n_out, 1)
            z = bn(z, training)
            z = tf.nn.relu(z)
            z = conv2d(z, n_out, 3)
            z = bn(z, training)

            # version 3
            #z = bn(z, training)
            #z = tf.nn.relu(z)
            #z = conv2d(z, n_out, 1)
            #z = bn(z, training)
            #z = tf.nn.relu(z)
            #z = conv2d(z, n_out, 3)
        return z

    def central_aggregation(inputs, n_out, training):
        with tf.variable_scope('central'):
            z = tf.concat(inputs, axis=-1)
            z = aggregation(z, n_out, training)
            # version 1
            #z = tf.nn.relu(z)

            # version 2
            z = tf.nn.relu(z)

            # version 3 (nothing)
        return z

    def local_aggregation(x, z, n_out, pos, training):
        with tf.variable_scope('local_{}'.format(pos)):
            y = tf.concat([x, z], axis=-1)
            y = x + aggregation(y, n_out, training)
            # version 1 (nothing)

            # version 2
            y = tf.nn.relu(y)

            # version 3
            #y = bn(y, training)
            #y = tf.nn.relu(y)
        return y

    with tf.variable_scope(scope):
        n_inputs = len(inputs)
        z = central_aggregation(inputs, nf * n_inputs // 4, training)
        outputs = [local_aggregation(x, z, nf, i, training)
                    for i, x in enumerate(inputs)]
    return outputs


def P_Net(
        input,
        label=None,
        bbox_target=None,
        landmark_target=None,
        training=True):

    def conv2d(x, nf, fs, act_fn=prelu):
        y = x
        y = slim.conv2d(y, nf, [fs, fs], 1, 'VALID', activation_fn=act_fn,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005))
        return y

    def maxpool(x, fs, stride):
        y = slim.max_pool2d(x, [fs, fs], stride, 'SAME')
        return y

    def block0(x, scope):
        with tf.variable_scope(scope):
            y = x
            y = conv2d(y, 10, 3)
            y = maxpool(y, 2, 2)
        return y

    def block1(x, scope):
        with tf.variable_scope(scope):
            y = x
            y = conv2d(y, 16, 3)
            y = conv2d(y, 32, 3)
        return y

    # start
    outputs = [input] * 3

    # block 0
    outputs = [block0(y, 'task_{}_block_0'.format(i)) for i, y in enumerate(outputs)]
    nf = outputs[0].get_shape().as_list()[-1]
    outputs = collaborative_block(outputs, nf, training, 'collaborative_block_0')

    # block 1
    outputs = [block1(y, 'task_{}_block_1'.format(i)) for i, y in enumerate(outputs)]
    nf = outputs[0].get_shape().as_list()[-1]
    outputs = collaborative_block(outputs, nf, training, 'collaborative_block_1')

    # output
    cls_pred, bbox_pred, landmark_pred = outputs

    with tf.variable_scope('classification'):
        cls_pred = conv2d(cls_pred, 2, 1, None)
        cls_pred = tf.nn.softmax(cls_pred)

    with tf.variable_scope('bounding_box'):
        bbox_pred = conv2d(bbox_pred, 4, 1, None)

    with tf.variable_scope('landmarks'):
        landmark_pred = conv2d(landmark_pred, 10, 1, None)

    if training:
        # classification: batch*2
        cls_prob = tf.squeeze(cls_pred, [1, 2], name='cls_prob')
        cls_loss = cls_ohem(cls_prob, label)
        accuracy = cal_accuracy(cls_prob, label)

        # bounding box: batch
        bbox_pred = tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')
        bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)

        # landmarks: batch*10
        landmark_pred = tf.squeeze(landmark_pred, [1, 2], name="landmark_pred")
        landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)

        # regularization
        L2_loss = tf.add_n(tf.losses.get_regularization_losses())

        return cls_loss, bbox_loss, landmark_loss, L2_loss, accuracy

    else: # test
        # when test: batch_size = 1
        cls_prob_test = tf.squeeze(cls_pred, axis=0)
        bbox_pred_test = tf.squeeze(bbox_pred, axis=0)
        landmark_pred_test = tf.squeeze(landmark_pred, axis=0)

        return cls_prob_test, bbox_pred_test, landmark_pred_test


def R_Net(
        inputs,
        label=None,
        bbox_target=None,
        landmark_target=None,
        training=True):

    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        net = slim.conv2d(inputs, 28, [3, 3], 1, scope="conv1")
        net = slim.max_pool2d(net, [3, 3], 2, 'SAME', scope="pool1")
        net = slim.conv2d(net, 48, [3, 3], 1, scope="conv2")
        net = slim.max_pool2d(net, [3, 3], 2, scope="pool2")
        net = slim.conv2d(net, 64, [2, 2], 1, scope="conv3")

        fc_flatten = slim.flatten(net)
        fc1 = slim.fully_connected(fc_flatten, 128, prelu, scope="fc1")

        # batch*2
        cls_prob = slim.fully_connected(fc1, 2, tf.nn.softmax, scope="cls_fc")

        # batch*4
        bbox_pred = slim.fully_connected(fc1, 4, None, scope="bbox_fc")

        # batch*10
        landmark_pred = slim.fully_connected(fc1, 10, None, scope="landmark_fc")

        # train
        if training:
            cls_loss = cls_ohem(cls_prob, label)
            bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
            accuracy = cal_accuracy(cls_prob, label)
            landmark_loss = landmark_ohem(
                landmark_pred, landmark_target, label)
            L2_loss = tf.add_n(tf.losses.get_regularization_losses())
            return cls_loss, bbox_loss, landmark_loss, L2_loss, accuracy
        else: # test
            return cls_prob, bbox_pred, landmark_pred


def O_Net(
        inputs,
        label=None,
        bbox_target=None,
        landmark_target=None,
        training=True):

    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):

        net = slim.conv2d(inputs, 32, [3, 3], 1, scope="conv1")
        net = slim.max_pool2d(net, [3, 3], 2, 'SAME', scope="pool1")
        net = slim.conv2d(net, 64, [3, 3], 1, scope="conv2")
        net = slim.max_pool2d(net, [3, 3], 2, scope="pool2")
        net = slim.conv2d(net, 64, [3, 3], 1, scope="conv3")
        net = slim.max_pool2d(net, [2, 2], 2, 'SAME', scope="pool3")
        net = slim.conv2d(net, 128, [2, 2], 1, scope="conv4")

        fc_flatten = slim.flatten(net)
        fc1 = slim.fully_connected(fc_flatten, 256, prelu, scope="fc1")

        # batch*2
        cls_prob = slim.fully_connected(fc1, 2, tf.nn.softmax, scope="cls_fc")

        # batch*4
        bbox_pred = slim.fully_connected(fc1, 4, None, scope="bbox_fc")

        # batch*10
        landmark_pred = slim.fully_connected(fc1, 10, None, scope="landmark_fc")

        # train
        if training:
            cls_loss = cls_ohem(cls_prob, label)
            bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
            accuracy = cal_accuracy(cls_prob, label)
            landmark_loss = landmark_ohem(
                landmark_pred, landmark_target, label)
            L2_loss = tf.add_n(tf.losses.get_regularization_losses())
            return cls_loss, bbox_loss, landmark_loss, L2_loss, accuracy
        else: # test
            return cls_prob, bbox_pred, landmark_pred
