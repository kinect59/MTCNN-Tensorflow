# coding:utf-8
import os
import random
import sys
import time
import tensorflow as tf
from argparse import ArgumentParser

from tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple


def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    # imaga_data:array to string
    # height:original image's height
    # width:original image's width
    # image_example dict contains image's info
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())


def get_input_output_filenames(input_dir, output_dir, net):
    if net == 'PNet':
        filenames = [
            (os.path.join(output_dir, 'train_PNet_landmark.txt'),
             os.path.join(output_dir, 'train_PNet_landmark.tfrecord'))]
    else:
        filenames = [
            (os.path.join(input_dir, 'landmark_{}_aug.txt'.format(input_dir)),
             os.path.join(output_dir, 'landmark_landmark.tfrecord')),
            (os.path.join(input_dir, 'neg_{}.txt'.format(input_dir)),
             os.path.join(output_dir, 'neg_landmark.tfrecord')),
            (os.path.join(input_dir, 'pos_{}.txt'.format(input_dir)),
             os.path.join(output_dir, 'pos_landmark.tfrecord')),
            (os.path.join(input_dir, 'part_{}.txt'.format(input_dir)),
             os.path.join(output_dir, 'part_landmark.tfrecord'))]

    return filenames


def get_dataset(dataset_file):
    dataset = []
    with open(dataset_file, 'r') as imagelist:

        for line in imagelist.readlines():
            info = line.strip().split(' ')
            data_example = dict()
            bbox = dict()
            data_example['filename'] = info[0]
            data_example['label'] = int(info[1])
            bbox['xmin'] = 0
            bbox['ymin'] = 0
            bbox['xmax'] = 0
            bbox['ymax'] = 0
            bbox['xlefteye'] = 0
            bbox['ylefteye'] = 0
            bbox['xrighteye'] = 0
            bbox['yrighteye'] = 0
            bbox['xnose'] = 0
            bbox['ynose'] = 0
            bbox['xleftmouth'] = 0
            bbox['yleftmouth'] = 0
            bbox['xrightmouth'] = 0
            bbox['yrightmouth'] = 0
            if len(info) == 6:
                bbox['xmin'] = float(info[2])
                bbox['ymin'] = float(info[3])
                bbox['xmax'] = float(info[4])
                bbox['ymax'] = float(info[5])
            if len(info) == 12:
                bbox['xlefteye'] = float(info[2])
                bbox['ylefteye'] = float(info[3])
                bbox['xrighteye'] = float(info[4])
                bbox['yrighteye'] = float(info[5])
                bbox['xnose'] = float(info[6])
                bbox['ynose'] = float(info[7])
                bbox['xleftmouth'] = float(info[8])
                bbox['yleftmouth'] = float(info[9])
                bbox['xrightmouth'] = float(info[10])
                bbox['yrightmouth'] = float(info[11])

            data_example['bbox'] = bbox
            dataset.append(data_example)

    return dataset


def run(input_dir, output_dir, net, shuffling=False):
    """Runs the conversion operation.

    Args:
      input_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """

    filenames = get_input_output_filenames(input_dir, output_dir, net)

    for txt_filename, tf_filename in filenames:

        if tf.gfile.Exists(tf_filename):
            print('Dataset files already exist. Exiting without re-creating them.')
            return

        dataset = get_dataset(txt_filename)

        if shuffling:
            tf_filename = tf_filename + '_shuffle'
            random.shuffle(dataset)

        # Write the data to tfrecord.
        print 'Start writing data to tfrecord...'
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            for i, image_example in enumerate(dataset):
                if i % 100000 == 0:
                    sys.stdout.write(
                        '\r>> Converting image %d/%d' %
                        (i + 1, len(dataset)))
                    sys.stdout.flush()
                filename = image_example['filename']
                _add_to_tfrecord(filename, image_example, tfrecord_writer)

        print '\nFinished writing data to tfrecord.'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('net', help='PNet, RNet or ONet')
    args = parser.parse_args()

    net = args.net

    if net == 'PNet':
        input_dir = '12'
    elif net == 'RNet':
        input_dir = '24'
    elif net == 'ONet':
        input_dir = '48'
    else:
        raise Exception("Invalid argument: {}".format(net))

    output_dir = os.path.join('imglists', net)

    run(input_dir, output_dir, net, shuffling=True)
