#!/usr/bin/env python
"""
Custom validation script that does NOT modify the original validate.py.
- Adds absolute MAE metrics (meters/mm)
- Opens PFM in binary mode
- Ensures validation result directory exists
- Defaults to local dataset/ckpt paths under /home/junho/dataset
"""
from __future__ import print_function

import os
import time
import sys
import math
import argparse
import numpy as np

import cv2
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# local imports
sys.path.append("../")
from tools.common import Notify
from preprocess import *
from model import *
from loss import *

"""
Usage (example):

python valid_custom.py \
  --regularization 3DCNNs \
  --validate_set dtu \
  --dtu_data_root /data/dtu/mvs_training/dtu \
  --view_num 5 \
  --max_w 1600 --max_h 1184 --max_d 256 \
  --pretrained_model_ckpt_path /data/ckpt/model.ckpt \
  --ckpt_step 150000 \
  --validation_result_path output/validation_results_custom.txt

"""

# -----------------------------
# Flags / Args
# -----------------------------
parser = argparse.ArgumentParser(description='Custom validation for MVSNet')
parser.add_argument('--dtu_data_root', type=str, default='/home/junho/dataset/dtu/mvs_training/dtu')
parser.add_argument('--validate_set', type=str, default='dtu', choices=['dtu','blendedmvs','eth3d'])
parser.add_argument('--view_num', type=int, default=5)
parser.add_argument('--max_d', type=int, default=256)
parser.add_argument('--max_w', type=int, default=1600)
parser.add_argument('--max_h', type=int, default=1184)
parser.add_argument('--sample_scale', type=float, default=0.25)
parser.add_argument('--interval_scale', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--inverse_depth', action='store_true')
parser.add_argument('--regularization', type=str, default='3DCNNs', choices=['3DCNNs','GRU'])
parser.add_argument('--pretrained_model_ckpt_path', type=str, default='/home/junho/dataset/tf_model/3DCNNs/model.ckpt')
parser.add_argument('--ckpt_step', type=int, default=85000)
parser.add_argument('--validation_result_path', type=str, default='output/validation_results_custom.txt')
args = parser.parse_args()

# Mirror args to FLAGS-like variables expected in imported helpers
class FLAGS_OBJ:
    pass
FLAGS = FLAGS_OBJ()
FLAGS.blendedmvs_data_root = '/data/BlendedMVS/dataset_low_res'
FLAGS.eth3d_data_root = '/data/eth3d/lowres/training/undistorted'
FLAGS.dtu_data_root = args.dtu_data_root
FLAGS.validate_set = args.validate_set
FLAGS.view_num = args.view_num
FLAGS.max_d = args.max_d
FLAGS.max_w = args.max_w
FLAGS.max_h = args.max_h
FLAGS.sample_scale = args.sample_scale
FLAGS.interval_scale = args.interval_scale
FLAGS.batch_size = args.batch_size
FLAGS.inverse_depth = args.inverse_depth
FLAGS.regularization = args.regularization
FLAGS.pretrained_model_ckpt_path = args.pretrained_model_ckpt_path
FLAGS.ckpt_step = args.ckpt_step
FLAGS.validation_result_path = args.validation_result_path
FLAGS.base_image_size = 32

# Ensure preprocess functions use our local FLAGS instead of tf.app.flags
import preprocess as preprocess_module
preprocess_module.FLAGS = FLAGS
import model as model_module
model_module.FLAGS = FLAGS

# -----------------------------
# Generator (PFM in binary mode)
# -----------------------------
class MVSGenerator:
    def __init__(self, sample_list, view_num):
        self.sample_list = sample_list
        self.view_num = view_num
    def __iter__(self):
        while True:
            for data in self.sample_list:
                images = []
                cams = []
                for view in range(self.view_num):
                    image = cv2.imread(data[2 * view])
                    cam = load_cam(open(data[2 * view + 1]), FLAGS.interval_scale)
                    cam[1, 3, 1] = (cam[1, 3, 3] - cam[1, 3, 0]) / FLAGS.max_d
                    cam[1, 3, 2] = FLAGS.max_d
                    images.append(image)
                    cams.append(cam)
                depth_image = load_pfm(open(data[2 * self.view_num], 'rb'))
                if FLAGS.validate_set == 'eth3d':
                    images, cams, depth_image = crop_mvs_input(images, cams, depth_image, max_w=FLAGS.max_w, max_h=FLAGS.max_h)
                    cams = scale_mvs_camera(cams, scale=FLAGS.sample_scale)
                    depth_image = scale_image(depth_image, scale=FLAGS.sample_scale)
                if FLAGS.validate_set == 'blendedmvs':
                    depth_image = scale_image(depth_image, scale=FLAGS.sample_scale)
                    cams = scale_mvs_camera(cams, scale=FLAGS.sample_scale)
                depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
                depth_end = cams[0][1, 3, 0] + (FLAGS.max_d - 2) * cams[0][1, 3, 1]
                depth_image = mask_depth_image(depth_image, depth_start, depth_end)
                images = np.stack(images, axis=0)
                cams = np.stack(cams, axis=0)
                yield (images, cams, depth_image)

# -----------------------------
# Build graph
# -----------------------------

def build_and_run(mvs_list):
    print('Validation sample number:', len(mvs_list))
    mvs_generator = iter(MVSGenerator(mvs_list, FLAGS.view_num))
    generator_data_type = (tf.float32, tf.float32, tf.float32)
    mvs_set = tf.data.Dataset.from_generator(lambda: mvs_generator, generator_data_type)
    mvs_set = mvs_set.batch(FLAGS.batch_size)
    mvs_set = mvs_set.prefetch(buffer_size=1)
    mvs_iterator = mvs_set.make_initializable_iterator()
    images, cams, depth_image = mvs_iterator.get_next()

    images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
    cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
    depth_image.set_shape(tf.TensorShape([None, None, None, 1]))
    depth_start = tf.reshape(tf.slice(cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    depth_interval = tf.reshape(tf.slice(cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    depth_num = tf.cast(tf.reshape(tf.slice(cams, [0, 0, 1, 3, 2], [1, 1, 1, 1, 1]), []), 'int32')
    if FLAGS.inverse_depth:
        depth_end = tf.reshape(tf.slice(cams, [0, 0, 1, 3, 3], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
    else:
        depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval

    # normalize images
    normalized_images = []
    for view in range(0, FLAGS.view_num):
        image = tf.squeeze(tf.slice(images, [0, view, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
        image = tf.image.per_image_standardization(image)
        normalized_images.append(image)
    images = tf.stack(normalized_images, axis=1)

    # inference
    if FLAGS.regularization == '3DCNNs':
        depth_map, prob_map = inference(images, cams, FLAGS.max_d, depth_start, depth_interval)
    elif FLAGS.regularization == 'GRU':
        depth_map, prob_map = inference_winner_take_all(images, cams, depth_num, depth_start, depth_end, reg_type='GRU', inverse_depth=FLAGS.inverse_depth)

    # losses/metrics
    if FLAGS.inverse_depth:
        interval = tf.ones_like(depth_interval)
        loss, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(depth_map, depth_image, interval)
    else:
        loss, less_one_accuracy, less_three_accuracy = mvsnet_regression_loss(depth_map, depth_image, depth_interval)
    # absolute MAE in meters (self-contained)
    def mae_abs_meters(y_true, y_pred):
        mask_true = tf.cast(tf.not_equal(y_true, 0.0), dtype=tf.float32)
        denom = tf.reduce_sum(mask_true, axis=[1, 2, 3]) + 1e-7
        masked_abs_error = tf.abs(mask_true * (y_true - y_pred))
        masked_mae = tf.reduce_sum(masked_abs_error, axis=[1, 2, 3])
        return tf.reduce_sum(masked_mae / denom)

    mae_abs = mae_abs_meters(depth_image, depth_map)

    # session
    init_op = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    ave_loss = 0.0
    ave_per1 = 0.0
    ave_per3 = 0.0
    ave_mae_abs = 0.0

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        # restore
        if FLAGS.pretrained_model_ckpt_path:
            restorer = tf.train.Saver(tf.global_variables())
            restorer.restore(sess, '-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)]))
            print(Notify.INFO, 'Restored from %s' % ('-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)])), Notify.ENDC)
        # iterate
        sess.run(mvs_iterator.initializer)
        for step in range(len(mvs_list)):
            start_time = time.time()
            try:
                out_loss, out_less_one, out_less_three, out_mae_abs = sess.run([loss, less_one_accuracy, less_three_accuracy, mae_abs])
            except tf.errors.OutOfRangeError:
                print('End of dataset')
                break
            duration = time.time() - start_time
            print(Notify.INFO, 'val %d: loss=%.3f, <1=%.3f, <3=%.3f, mae=%.3f m (%.3f s/step)' % (step, out_loss, out_less_one, out_less_three, out_mae_abs, duration), Notify.ENDC)
            ave_loss += out_loss
            ave_per1 += out_less_one
            ave_per3 += out_less_three
            ave_mae_abs += out_mae_abs

        n = float(len(mvs_list))
        ave_loss /= n
        ave_per1 /= n
        ave_per3 /= n
        ave_mae_abs /= n
        print('ave_loss', ave_loss)
        print('ave_per1', ave_per1)
        print('ave_per3', ave_per3)
        print('ave_mae_meters', ave_mae_abs)
        print('ave_mae_millimeters', ave_mae_abs * 1000.0)

        # ensure output dir
        vr_dir = os.path.dirname(FLAGS.validation_result_path)
        if vr_dir and not os.path.exists(vr_dir):
            os.makedirs(vr_dir, exist_ok=True)
        with open(FLAGS.validation_result_path, 'a') as f:
            f.write('ckpt %d | L1(step)=%.6f, <1=%.6f, <3=%.6f, MAE(m)=%.6f, MAE(mm)=%.2f\n' % (
                int(FLAGS.ckpt_step), float(ave_loss), float(ave_per1), float(ave_per3), float(ave_mae_abs), float(ave_mae_abs*1000.0)))

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    print('Custom Validating MVSNet with %d views' % FLAGS.view_num)
    # Avoid absl.flags parsing argparse flags by sanitizing argv and pre-parsing with empty args
    try:
        sys.argv = [sys.argv[0]]
        from absl import flags as absl_flags
        absl_flags.FLAGS(['valid_custom'])
    except Exception:
        pass
    # make sample list
    if FLAGS.validate_set == 'blendedmvs':
        sample_list = gen_blendedmvs_path(FLAGS.blendedmvs_data_root, mode='validation')
    elif FLAGS.validate_set == 'eth3d':
        sample_list = gen_eth3d_path(FLAGS.eth3d_data_root, mode='validation')
    elif FLAGS.validate_set == 'dtu':
        sample_list = gen_dtu_resized_path(FLAGS.dtu_data_root, mode='validation')
    else:
        raise ValueError('Unknown validate_set: %s' % FLAGS.validate_set)
    build_and_run(sample_list)
