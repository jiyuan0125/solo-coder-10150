import sys
import os
import time
import math
import torch
import numpy as np
from torch.autograd import Variable

import itertools
import struct  # get_image_size
try:
    import imghdr  # get_image_size
except ModuleNotFoundError:
    imghdr = None  # removed in Python 3.13+

from tool import utils
from tool.utils_iou import validate_iou_type


def bbox_ious(boxes1, boxes2, x1y1x2y2=True, iou_type='iou'):
    iou_type = validate_iou_type(iou_type)
    if x1y1x2y2:
        x1, y1, x2, y2 = boxes1[0], boxes1[1], boxes1[2], boxes1[3]
        x1b, y1b, x2b, y2b = boxes2[0], boxes2[1], boxes2[2], boxes2[3]
        w1 = x2 - x1
        h1 = y2 - y1
        w2 = x2b - x1b
        h2 = y2b - y1b
        cx1 = (x1 + x2) / 2.0
        cy1 = (y1 + y2) / 2.0
        cx2 = (x1b + x2b) / 2.0
        cy2 = (y1b + y2b) / 2.0
        mx = torch.min(x1, x1b)
        Mx = torch.max(x2, x2b)
        my = torch.min(y1, y1b)
        My = torch.max(y2, y2b)
    else:
        cx1, cy1, w1, h1 = boxes1[0], boxes1[1], boxes1[2], boxes1[3]
        cx2, cy2, w2, h2 = boxes2[0], boxes2[1], boxes2[2], boxes2[3]
        x1 = cx1 - w1 / 2.0
        y1 = cy1 - h1 / 2.0
        x2 = cx1 + w1 / 2.0
        y2 = cy1 + h1 / 2.0
        x1b = cx2 - w2 / 2.0
        y1b = cy2 - h2 / 2.0
        x2b = cx2 + w2 / 2.0
        y2b = cy2 + h2 / 2.0
        mx = torch.min(x1, x1b)
        Mx = torch.max(x2, x2b)
        my = torch.min(y1, y1b)
        My = torch.max(y2, y2b)

    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    iou = carea / uarea.clamp(min=1e-16)

    if iou_type == 'iou':
        return iou

    c_area = uw * uh + 1e-16

    if iou_type == 'giou':
        giou = iou - (c_area - uarea) / c_area
        return giou

    centre_dist2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    diag_len2 = uw ** 2 + uh ** 2 + 1e-16
    diou = iou - centre_dist2 / diag_len2

    if iou_type == 'diou':
        return diou

    if iou_type == 'ciou':
        v = (4.0 / (math.pi ** 2)) * (torch.atan(w1 / (h1 + 1e-16)) - torch.atan(w2 / (h2 + 1e-16))) ** 2
        with torch.no_grad():
            denom = 1.0 - iou + v + 1e-16
            alpha_raw = v / denom
            alpha = torch.where(iou >= 0.5, alpha_raw, torch.zeros_like(alpha_raw))
        ciou = diou - alpha * v
        return ciou

    return iou


def get_region_boxes(boxes_and_confs):

    # print('Getting boxes from boxes and confs ...')

    boxes_list = []
    confs_list = []

    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])

    # boxes: [batch, num1 + num2 + num3, 1, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)
        
    return [boxes, confs]


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)



def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1, iou_type='iou'):
    model.eval()
    with torch.no_grad():
        t0 = time.time()

        if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
            img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        elif type(img) == np.ndarray and len(img.shape) == 4:
            img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
        else:
            print("unknow image type")
            exit(-1)

        if use_cuda:
            img = img.cuda()
        img = torch.autograd.Variable(img)

        t1 = time.time()

        output = model(img)

        t2 = time.time()

        print('-----------------------------------')
        print('           Preprocess : %f' % (t1 - t0))
        print('      Model Inference : %f' % (t2 - t1))
        print('-----------------------------------')

        return utils.post_processing(img, conf_thresh, nms_thresh, output, iou_type=iou_type)

