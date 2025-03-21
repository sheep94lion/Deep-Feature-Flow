# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xizhou Zhu, Yi Li, Haochen Zhang
# --------------------------------------------------------

#%%
cur_path = "/home/yizhao/Code/Deep-Feature-Flow/dff_rfcn/"
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
key_sym = sym_instance.get_key_test_symbol(config)
cur_sym = sym_instance.get_cur_test_symbol(config)

cur_sym_flow = sym_instance.get_cur_test_flow_symbol(config)
cur_sym_prop = sym_instance.get_cur_test_prop_symbol(config)
cur_sym_rpn = sym_instance.get_cur_test_rpn_symbol(config)

key_sym_feat = sym_instance.get_key_test_feat_symbol(config)
key_sym_rpn = sym_instance.get_key_test_rpn_symbol(config)

# cur_sym.save("dff_cur_sym2.json")
# return

# get the input shape to rpn_inv_normalize



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

# load demo data
image_names = sorted(glob.glob(cur_path + '/../demo/ILSVRC2015_val_00007010/*.JPEG'))
# print image_names
# return
output_dir = cur_path + '/../demo/rfcn_dff_cpu/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
key_frame_interval = 10

arg_params, aux_params = load_param(cur_path + model, 0, process=True)

# Module
mod_cur_flow = mx.mod.Module(symbol=cur_sym_flow, context=ctx, data_names=('data', 'data_key'), label_names=None)
mod_cur_prop = mx.mod.Module(symbol=cur_sym_prop, context=ctx, data_names=('flow', 'scale_map', 'feat_key'), label_names=None)
mod_cur_rpn = mx.mod.Module(symbol=cur_sym_rpn, context=ctx, data_names=('conv_feat', 'im_info'), label_names=None)
mod_cur_flow.bind(for_training=False, data_shapes=[('data', (1, 3, 562, 1000)), ('data_key', (1, 3, 562, 1000))])
# print mod_flow.output_shapes
mod_cur_prop.bind(for_training=False, data_shapes=[('flow', (1, 2, 36, 63)), ('scale_map', (1, 1024, 36, 63)), ('feat_key', (1, 1024, 36, 63))])
# print mod_prop.output_shapes
mod_cur_rpn.bind(for_training=False, data_shapes=[('conv_feat', (1, 1024, 36, 63)), ('im_info', (1, 3))])
# print mod_rpn.output_shapes

mod_key_feat = mx.mod.Module(symbol=key_sym_feat, context=ctx, data_names=('data',), label_names=None)
mod_key_rpn = mx.mod.Module(symbol=key_sym_rpn, context=ctx, data_names=('conv_feat', 'im_info'), label_names=None)
mod_key_feat.bind(for_training=False, data_shapes=[('data', (1, 3, 562, 1000))])
mod_key_rpn.bind(for_training=False, data_shapes=[('conv_feat', (1, 1024, 36, 63)), ('im_info', (1, 3))])

mod_key = mx.mod.Module(symbol=key_sym, context=ctx, data_names=('data', 'im_info', 'data_key', 'feat_key'), label_names=None)
mod_key.bind(for_training=False, data_shapes=[('data', (1, 3, 562, 1000)), ('im_info', (1, 3)), ('data_key', (1, 3, 562, 1000)), ('feat_key', (1, 1024, 36, 63))])

mod_cur_flow.set_params(arg_params, aux_params)
mod_cur_prop.set_params(arg_params, aux_params)
mod_cur_rpn.set_params(arg_params, aux_params)

mod_key_feat.set_params(arg_params, aux_params)
mod_key_rpn.set_params(arg_params, aux_params)

mod_key.set_params(arg_params, aux_params)

#%%
data = []
key_im_tensor = None
for idx, im_name in enumerate(image_names):
    assert os.path.exists(im_name), ('%s does not exist'.format(im_name))
    im = cv2.imread(im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    target_size = config.SCALES[0][0]
    max_size = config.SCALES[0][1]
    im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
    im_tensor = transform(im, config.network.PIXEL_MEANS)
    im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
    # print(im_info.shape)
    # print im_info
    if idx % key_frame_interval == 0:
        key_im_tensor = im_tensor
    data.append({'data': im_tensor, 'im_info': im_info, 'data_key': key_im_tensor, 'feat_key': np.zeros((1,config.network.DFF_FEAT_DIM,36,63))})


# get predictor
# data_names = ['data', 'im_info', 'data_key', 'feat_key']
# label_names = []
# data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
# max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES]))),
#                     ('data_key', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES]))),]]
# provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
# provide_label = [None for i in xrange(len(data))]
# arg_params, aux_params = load_param(cur_path + model, 0, process=True)
# key_predictor = Predictor(key_sym, data_names, label_names,
#                         context=[ctx], max_data_shapes=max_data_shape,
#                         provide_data=provide_data, provide_label=provide_label,
#                         arg_params=arg_params, aux_params=aux_params)
# cur_predictor = Predictor(cur_sym, data_names, label_names,
#                         context=[ctx], max_data_shapes=max_data_shape,
#                         provide_data=provide_data, provide_label=provide_label,
#                         arg_params=arg_params, aux_params=aux_params)

if device_name == 'cpu':
    nms = cpu_nms_wrapper(config.TEST.NMS)
else:
    nms = gpu_nms_wrapper(config.TEST.NMS, 0)


# print data[0]['data'].shape

#%%
from collections import namedtuple
BatchKeyFeat = namedtuple('BatchKeyFeat', ['data'])
BatchKeyRpn = namedtuple('BatchKeyRpn', ['conv_feat', 'im_info'])
BatchKey = namedtuple('BatchKey', ['data', 'im_info', 'data_key', 'feat_key'])

# lists to store running time.
time_list_key_feat = []
time_list_key_rpn = []
time_list_key = []

time_list_cur_flow = []
time_list_cur_prop = []
time_list_cur_rpn = []
time_list_cur = []

for idx, im_name in enumerate(image_names):
    im_shape = data[idx]['data'].shape
    if idx % key_frame_interval == 0:
        tic()
        data_batch_key_feat = mx.io.DataBatch(data=[mx.nd.array(data[idx][data_name]) for data_name in ['data']], provide_data=[mx.io.DataDesc('data', (1, 3, 562, 1000))])
        mod_key_feat.forward(data_batch_key_feat)
        conv_feat = mod_key_feat.get_outputs()[0]
        if conv_feat[0][0][0][0] == 0.3:
            nothing = 0
        time_key_feat = toc()
        time_list_key_feat.append(time_key_feat)

        # print type(conv_feat)
        # print 'As Parts: conv_feat: ', mod_key_feat.get_outputs()
        tic()
        data_batch_key_rpn = mx.io.DataBatch(data=[conv_feat, mx.nd.array(data[idx]['im_info'])], provide_data=[mx.io.DataDesc('conv_feat', (1, 1024, 36, 63)), mx.io.DataDesc('im_info', (1, 3))])
        mod_key_rpn.forward(data_batch_key_rpn)
        rois_output, cls_prob, bbox_pred_c = mod_key_rpn.get_outputs()
        feat_key = conv_feat
    else:
        tic()
        data_batch_cur_flow = mx.io.DataBatch(data=[mx.nd.array(data[idx][data_name]) for data_name in ['data', 'data_key']], provide_data=[mx.io.DataDesc('data', (1, 3, 562, 1000)), mx.io.DataDesc('data_key', (1, 3, 562, 1000))])
        mod_cur_flow.forward(data_batch_cur_flow)
        flow, scale_map = mod_cur_flow.get_outputs()
        if flow[0][0][0][0] == 0.3:
            nothing = 1
        if scale_map[0][0][0][0] == 0.3:
            nothing = 1
        time_cur_flow = toc()
        time_list_cur_flow.append(time_cur_flow)
        
        # ('flow', (1, 2, 36, 63)), ('scale_map', (1, 1024, 36, 63)), ('feat_key', (1, 1024, 36, 63))
        # Cur frame feature map propagation
        tic()
        data_batch_cur_prop = mx.io.DataBatch(data=[flow, scale_map, feat_key], provide_data=[mx.io.DataDesc('flow', (1, 2, 36, 63)), mx.io.DataDesc('scale_map', (1, 1024, 36, 63)), mx.io.DataDesc('feat_key', (1, 1024, 36, 63))])
        mod_cur_prop.forward(data_batch_cur_prop)
        conv_feat = mod_cur_prop.get_outputs()[0]
        if conv_feat[0][0][0][0] == 0.3:
            nothing = 1
        time_cur_prop = toc()
        time_list_cur_prop.append(time_cur_prop)

        tic()
        data_batch_cur_rpn = mx.io.DataBatch(data=[conv_feat, mx.nd.array(data[idx]['im_info'])], provide_data=[mx.io.DataDesc('conv_feat', (1, 1024, 36, 63)), mx.io.DataDesc('im_info', (1, 3))])
        mod_cur_rpn.forward(data_batch_cur_rpn)
        rois_output, cls_prob, bbox_pred_c = mod_cur_rpn.get_outputs()

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
    if idx % key_frame_interval == 0:
        time_key_rpn = toc()
        time_list_key_rpn.append(time_key_rpn)
        time_key = time_key_feat + time_key_rpn
        time_list_key.append(time_key_feat + time_key_rpn)
        print "Process key frame {}, Feature extraction: {:.4f}s, Task network: {:.4f}s, Totally: {:.4f}s.".format(im_name, time_key_feat, time_key_rpn, time_key)
    else:
        time_cur_rpn = toc()
        time_list_cur_rpn.append(time_cur_rpn)
        time_cur = time_cur_flow + time_cur_prop + time_cur_rpn
        time_list_cur.append(time_cur_flow + time_cur_prop + time_cur_rpn)
        print "Process non-key frame {}, flow: {:.4f}s, prop: {:.4f}s, rpn: {:.4f}s, Totally: {:.4f}s.".format(im_name, time_cur_flow, time_cur_prop, time_cur_rpn, time_cur)

    # visualize
    im = cv2.imread(im_name)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # show_boxes(im, dets_nms, classes, 1)
    out_im = draw_boxes(im, dets_nms, classes, 1)
    _, filename = os.path.split(im_name)
    output_dir = "/home/yizhao/Code/Deep-Feature-Flow/demo/rfcn_dff_components/"
    cv2.imwrite(output_dir + filename,out_im)
print "Average running time of key frame, feat: {:.4f}s, rpn: {:.4f}s, total {:.4f}s".format(np.average(time_list_key_feat), np.average(time_list_key_rpn), np.average(time_list_key))
print "Average running time of non-key frame, flow: {:.4f}s, prop: {:.4f}s, rpn: {:.4f}s, total: {:.4f}s.".format(np.average(time_list_cur_flow), np.average(time_list_cur_prop), np.average(time_list_cur_rpn), np.average(time_list_cur))
exit(0)

    # data_batch_key = mx.io.DataBatch(data=[mx.nd.array(data[idx]['data']), mx.nd.array(data[idx]['im_info']), mx.nd.array(data[idx]['data_key']), mx.nd.array(data[idx]['feat_key'])], label=[], pad=0, index=0,provide_data=[mx.io.DataDesc('data', (1, 3, 562, 1000)), mx.io.DataDesc('im_info', (1, 3)), mx.io.DataDesc('data_key', (1, 3, 562, 1000)), mx.io.DataDesc('feat_key', (1, 1024, 36, 63))])
    # mod_key.forward(data_batch_key)
    # print 'As Whole: ', mod_key.get_outputs()
    # print inter_conv_feat.shape
    # mod_key_rpn.forward(BatchKeyRpn([inter_conv_feat], [data[idx]['im_info']]))
    # rois, cls_prob, bbox_pred = mod_key_rpn.get_outputs()
    # print rois, cls_prob, bbox_pred
    # mod_key.forward(BatchKey(mx.nd.array([data[idx]['data']]), mx.nd.array([data[idx]['im_info']]), mx.nd.array([data[idx]['data_key']]), mx.nd.array([data[idx]['feat_key']])))
    # print mod_key.get_outputs()
# exit(0)

#%%
# warm up
for j in xrange(2):
    data_batch = mx.io.DataBatch(data=[data[j]], label=[], pad=0, index=0,
                                    provide_data=[[(k, v.shape) for k, v in zip(data_names, data[j])]],
                                    provide_label=[None])
    scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
    if j % key_frame_interval == 0:
        scores, boxes, data_dict, feat = im_detect(key_predictor, data_batch, data_names, scales, config)
    else:
        data_batch.data[0][-1] = feat
        data_batch.provide_data[0][-1] = ('feat_key', feat.shape)
        scores, boxes, data_dict, _ = im_detect(cur_predictor, data_batch, data_names, scales, config)

#%%
mx.nd.save('feat_key_list', [feat])
feat_np = feat.asnumpy()
print feat_np.shape
feat_np.tofile('feat_key.binary')
h5f = h5py.File('feat_key.h5', 'w')
print h5f.create_dataset('dataset_1', data=feat_np)
h5f.close()
# exit(0)

#%%
print "warmup done"
# test
time = 0
count = 0
cur_frame_time_list = []
key_frame_time_list = []
for idx, im_name in enumerate(image_names):
    data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                    provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                    provide_label=[None])
    scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

    tic()
    if idx % key_frame_interval == 0:
        scores, boxes, data_dict, feat = im_detect(key_predictor, data_batch, data_names, scales, config)
    else:
        data_batch.data[0][-1] = feat
        data_batch.provide_data[0][-1] = ('feat_key', feat.shape)
        scores, boxes, data_dict, _ = im_detect(cur_predictor, data_batch, data_names, scales, config)
    current_time = toc()
    time += current_time
    count += 1
    print 'testing {} current: {:.4f}s average: {:.4f}s'.format(im_name, current_time , time/count)
    if idx % key_frame_interval == 0:
        key_frame_time_list.append(current_time)
    else:
        cur_frame_time_list.append(current_time)

    boxes = boxes[0].astype('f')
    scores = scores[0].astype('f')
    dets_nms = []
    for j in range(1, scores.shape[1]):
        cls_scores = scores[:, j, np.newaxis]
        cls_boxes = boxes[:, 4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores))
        keep = nms(cls_dets)
        cls_dets = cls_dets[keep, :]
        cls_dets = cls_dets[cls_dets[:, -1] > 0.7, :]
        dets_nms.append(cls_dets)
    # visualize
    im = cv2.imread(im_name)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # show_boxes(im, dets_nms, classes, 1)
    out_im = draw_boxes(im, dets_nms, classes, 1)
    _, filename = os.path.split(im_name)
    cv2.imwrite(output_dir + filename,out_im)
print 'Average running time for key frame: {:.4f}s, non-key frame: {:.4f}s'.format(np.average(key_frame_time_list), np.average(cur_frame_time_list))

print 'done'

