#%%
cur_path = "/home/yizhao/Code/Deep-Feature-Flow/ddff_client/"
import sys
sys.path.append(cur_path)
import _init_paths
import h5py
import argparse
import os
import glob
import sys
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
update_config(cur_path + '/../experiments/dff_rfcn/cfgs/dff_rfcn_vid_demo.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import im_detect, Predictor, bbox_pred, clip_boxes
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes, draw_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper


# get symbol
pprint.pprint(config)
device_name = 'cpu'
if device_name == 'cpu':
    ctx = mx.cpu()
else:
    ctx = mx.gpu()
config.symbol = 'resnet_v1_101_flownet_rfcn'
model = '/../model/rfcn_dff_flownet_vid'
sym_instance = eval(config.symbol + '.' + config.symbol)()

# get the network symbol to process current frame
cur_sym = sym_instance.get_cur_test_symbol(config)

# set up class names
num_classes = 31
classes = ['airplane', 'antelope', 'bear', 'bicycle',
            'bird', 'bus', 'car', 'cattle',
            'dog', 'domestic_cat', 'elephant', 'fox',
            'giant_panda', 'hamster', 'horse', 'lion',
            'lizard', 'monkey', 'motorcycle', 'rabbit',
            'red_panda', 'sheep', 'snake', 'squirrel',
            'tiger', 'train', 'turtle', 'watercraft',
            'whale', 'zebra']

output_dir = cur_path + '/../demo/ddff_client/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# key_frame_interval = 10

arg_params, aux_params = load_param(cur_path + model, 0, process=True)

# Module
mod_cur = mx.mod.Module(symbol=cur_sym, context=ctx, data_names=('data', 'im_info', 'data_key', 'feat_key'), label_names=None)
mod_cur.bind(for_training=False, data_shapes=[('data', (1, 3, 562, 1000)), ('im_info', (1, 3)), ('data_key', (1, 3, 562, 1000)), ('feat_key', (1, 1024, 36, 63))])
mod_cur.set_params(arg_params, aux_params)

if device_name == 'cpu':
    nms = cpu_nms_wrapper(config.TEST.NMS)
else:
    nms = gpu_nms_wrapper(config.TEST.NMS, 0)

# Load current frame and key frame from image file
key_frame_path = "/home/yizhao/Code/Deep-Feature-Flow/demo/ILSVRC2015_val_00007010/000000.JPEG"
cur_frame_path = "/home/yizhao/Code/Deep-Feature-Flow/demo/ILSVRC2015_val_00007010/000009.JPEG"

key_frame = cv2.imread(key_frame_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
target_size = config.SCALES[0][0]
max_size = config.SCALES[0][1]
key_frame, im_scale = resize(key_frame, target_size, max_size, stride=config.network.IMAGE_STRIDE)
key_im_tensor = transform(key_frame, config.network.PIXEL_MEANS)
key_im_info = np.array([[key_im_tensor.shape[2], key_im_tensor.shape[3], im_scale]], dtype=np.float32)

cur_frame = cv2.imread(cur_frame_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
target_size = config.SCALES[0][0]
max_size = config.SCALES[0][1]
cur_frame, im_scale = resize(cur_frame, target_size, max_size, stride=config.network.IMAGE_STRIDE)
cur_im_tensor = transform(cur_frame, config.network.PIXEL_MEANS)
im_shape = cur_im_tensor.shape
cur_im_info = np.array([[cur_im_tensor.shape[2], cur_im_tensor.shape[3], im_scale]], dtype=np.float32)

# Load feature map of key frame from file
feat_key = np.fromfile("/home/yizhao/Code/Deep-Feature-Flow/feat_key.binary", dtype='f4')
feat_key = feat_key.reshape(1, 1024, 36, 63)
#%%
# Network inference
data_batch_cur = mx.io.DataBatch(data=[mx.nd.array(cur_im_tensor), mx.nd.array(cur_im_info), mx.nd.array(key_im_tensor), mx.nd.array(feat_key)], provide_data=[mx.io.DataDesc('data', (1, 3, 562, 1000)), mx.io.DataDesc('im_info', (1, 3)), mx.io.DataDesc('data_key', (1, 3, 562, 1000)), mx.io.DataDesc('feat_key', (1, 1024, 36, 63))])
mod_cur.forward(data_batch_cur)
rois_output, cls_prob, bbox_pred_c = mod_cur.get_outputs()

# process outputs
rois = rois_output.asnumpy()[:, 1:]
scores = cls_prob.asnumpy()[0]
bbox_deltas = bbox_pred_c.asnumpy()[0]

# post processing
pred_boxes = bbox_pred(rois, bbox_deltas)
pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])
scale = 1000.0 / 1280.0
# we used scaled image & roi to train, so it is necessary to transform them back
pred_boxes = pred_boxes / scale
scores = scores.astype('f')
pred_boxes = pred_boxes.astype('f')

dets_nms = []
for j in range(1, scores.shape[1]):
    cls_scores = scores[:, j, np.newaxis]
    cls_boxes = pred_boxes[:, 4:8] if config.CLASS_AGNOSTIC else pred_boxes[:, j * 4:(j + 1) * 4]
    cls_dets = np.hstack((cls_boxes, cls_scores))
    keep = nms(cls_dets)
    cls_dets = cls_dets[keep, :]
    cls_dets = cls_dets[cls_dets[:, -1] > 0.7, :]
    dets_nms.append(cls_dets)

# visualize
im = cv2.imread(cur_frame_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# show_boxes(im, dets_nms, classes, 1)
out_im = draw_boxes(im, dets_nms, classes, 1)
_, filename = os.path.split(cur_frame_path)
output_dir = "/home/yizhao/Code/Deep-Feature-Flow/demo/ddff_client/"
cv2.imwrite(output_dir + filename,out_im)
cv2.imshow(cur_frame_path, out_im)
cv2.waitKey(0)

print "done"