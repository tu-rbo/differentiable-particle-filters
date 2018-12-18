import numpy as np
import matplotlib.pyplot as plt
import os
import math
import glob
from time import time
from PIL import Image
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def load_image(img_file):
    tmp = Image.open(img_file)
    tmp = tmp.resize(size=(1280, 384))

    return np.asarray(tmp)

def pad(tensor, num=1):
    """
    Pads the given tensor along the height and width dimensions with `num` 0s on each side
    """
    return tf.pad(tensor, [[0, 0], [num, num], [num, num], [0, 0]], "CONSTANT")

def LeakyReLU(x, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1.0 + leak)
        f2 = 0.5 * (1.0 - leak)
        return f1 * x + f2 * abs(x)

# loading all sequences for KITTI
def store_kitti_sequences_as_tf_record(sequence_list=None):

    path = "/mnt/StorageDevice/KITTI/original_dataset/dataset/sequences"

    print('Loading KITTI DATA')

    if sequence_list is None:
        sequence_list = list(range(11))

    print('Cache not found, loading from KITTI_dataset')

    image_seq_1_full_path = ["{}/{:02d}/image_2".format(path, x) for x in sequence_list]
    image_seq_2_full_path = ["{}/{:02d}/image_3".format(path, x) for x in sequence_list]

    # Extract original image and difference image
    for i in sequence_list:
        input_image_file = []

        for name in glob.glob('{}/*.png'.format(image_seq_1_full_path[i])):
            input_image_file = input_image_file + [name]
        for name in glob.glob('{}/*.png'.format(image_seq_2_full_path[i])):
            input_image_file = input_image_file + [name]

        input_image_file.sort()
    # print (input_image_file)

        oxts_seq_1 = ["%.2d_image1.txt" % i]
        oxts_seq_1 = oxts_seq_1 + ["%.2d_image2.txt" % i]
        oxts_seq_1.sort()
        oxts_seq_1_full_path = ["{}/{}".format(path, x) for x in oxts_seq_1]
        output_oxts_file = oxts_seq_1_full_path

        path_to_save = "/mnt/StorageDevice/KITTI"
        tfrecords_filename = "kitti.tfrecords"

        writer = tf.python_io.TFRecordWriter("{}/kitti_{}.tfrecords".format(path_to_save, i))

        s = []
        for j in range(len(output_oxts_file)):
    #   # load text file
            with open(output_oxts_file[j], 'r') as f:
                s.append(np.loadtxt(f)) #Add all required states to tf records
        s = np.concatenate(s)
        print (np.shape(s))
    #     start = 0 if ii == 0 else seq_num[ii - 1]
    #
    #     x = tmp[:, 11]
    #     y = -tmp[:, 3]
    #     theta = -np.arctan2(-tmp[:, 8], tmp[:, 10])
    #     s[start:seq_num[ii], 0] = x[1:]  # x
    #     s[start:seq_num[ii], 1] = y[1:]  # y
    #     s[start:seq_num[ii], 2] = theta[1:]  # angle
    #     s[start:seq_num[ii], 3] = np.sqrt((y[1:] - y[:-1]) ** 2 + (x[1:] - x[:-1]) ** 2) / 0.103  # forward vel
    #     s[start:seq_num[ii], 4] = wrap_angle(theta[1:] - theta[:-1]) / 0.103  # angular vel

        count = 0
        for img_path in input_image_file:
            img = load_image(img_path)
            height = img.shape[0]
            width = img.shape[1]


            img_raw = img.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'image_raw': _bytes_feature(img_raw),
                'x': _float_feature(s[count, 11]),
                'y': _float_feature(-s[count, 3]),
                'theta': _float_feature(-np.arctan2(-s[count, 8], s[count, 10]))}))

            writer.write(example.SerializeToString())
        count += 1
        print (count)

def test_tfrecord():

    path_to_save = "/mnt/StorageDevice/KITTI"
    tfrecords_filename = "kitti.tfrecords"


    reconstructed_images = []

    record_iterator = tf.python_io.tf_record_iterator(path="{}/{}".format(path_to_save, tfrecords_filename))

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['height']
                     .int64_list
                     .value[0])

        width = int(example.features.feature['width']
                    .int64_list
                    .value[0])

        img_string = (example.features.feature['image_raw']
                      .bytes_list
                      .value[0])


        img_1d = np.fromstring(img_string, dtype=np.uint8)
        reconstructed_img = img_1d.reshape((height, width, -1))

        print(height)
    # oxts_seq_1 = ["%.2d_image1.txt" % i for i in sequence_list]
    # oxts_seq_1 = oxts_seq_1 + ["%.2d_image2.txt" % i for i in sequence_list]
    # oxts_seq_1.sort()
    # oxts_seq_1_full_path = ["{}/{}".format(path, x) for x in oxts_seq_1]
    # output_oxts_file = oxts_seq_1_full_path
    #
    # sequence_starts_ends = [[0, 4540], [0, 1100], [0, 4660], [0, 800], [0, 270], [0, 2760], [0, 1100], [0, 1100], [1100, 5170], [0, 1590],
    #  [0, 1200]]
    # data_values = np.array([sequence_starts_ends[i] for i in sequence_list])
    # seq_num = np.zeros((2*data_values.shape[0],))
    # weights = np.zeros((2*data_values.shape[0],))
    #
    # for ii in range(data_values.shape[0]):
    #     if ii == 0:
    #         seq_num[0] = data_values[ii,1] - data_values[ii,0]
    #         seq_num[1] = seq_num[0] + data_values[ii,1] - data_values[ii,0]
    #         weights[0] = weights[1] = data_values[ii,1] - data_values[ii,0]
    #     else:
    #         seq_num[2*ii] = seq_num[2*ii-1] + data_values[ii, 1] - data_values[ii, 0]
    #         seq_num[2*ii+1] = seq_num[2*ii] + data_values[ii, 1] - data_values[ii, 0]
    #         weights[2*ii] = weights[2*ii+1] = data_values[ii, 1] - data_values[ii, 0]
    #
    # # seq_num is an array of the cumulative sequence lengths, e.g. [100, 300, 350] for sequences of length 100, 200, 50
    # seq_num = seq_num.astype(int)
    # weights = weights/seq_num[-1]
    # print(seq_num, weights)
    #
    # o = np.zeros((seq_num[-1], 50, 150, 6))
    # count = 0
    # # for all sequences
    # for ii in range(len(seq_num)):
    #     # find out the start and end of the current sequence
    #     if ii == 0:
    #         start = 1
    #     else:
    #         start = seq_num[ii-1]+ii+1
    #
    #     # load first image
    #     prev_image = load_image(input_image_file[start-1])
    #     # for all time steps
    #     for jj in range(start, seq_num[ii]+ii+1):
    #         # load next image
    #         cur_image = load_image(input_image_file[jj])
    #         # observation from current and last image
    #         o[count, :, :, :] = image_input(cur_image, prev_image)
    #         prev_image = cur_image
    #         count += 1
    #
    # a = np.zeros((seq_num[-1], 3))
    # s = np.zeros((seq_num[-1], 5))
    # for ii in range(len(output_oxts_file)):
    #
    #     # load text file
    #     with open(output_oxts_file[ii], 'r') as f:
    #         tmp = np.loadtxt(f)
    #
    #     start = 0 if ii == 0 else seq_num[ii-1]

if __name__ == '__main__':
    store_kitti_sequences_as_tf_record()
    # test_tfrecord()