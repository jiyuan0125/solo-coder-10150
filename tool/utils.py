import sys
import os
import time
import math
import numpy as np

import itertools
import struct  # get_image_size
try:
    import imghdr  # get_image_size
except ModuleNotFoundError:
    imghdr = None  # removed in Python 3.13+


def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1.)


def softmax(x):
    x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    x = x / np.expand_dims(x.sum(axis=1), axis=1)
    return x


from tool.utils_iou import validate_iou_type


def bbox_iou(box1, box2, x1y1x2y2=True, iou_type='iou'):

    iou_type = validate_iou_type(iou_type)

    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
        cx1 = (box1[0] + box1[2]) / 2.0
        cy1 = (box1[1] + box1[3]) / 2.0
        cx2 = (box2[0] + box2[2]) / 2.0
        cy2 = (box2[1] + box2[3]) / 2.0
    else:
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
        cx1 = box1[0]
        cy1 = box1[1]
        cx2 = box2[0]
        cy2 = box2[1]

        mx = min(box1[0] - w1 / 2.0, box2[0] - w2 / 2.0)
        Mx = max(box1[0] + w1 / 2.0, box2[0] + w2 / 2.0)
        my = min(box1[1] - h1 / 2.0, box2[1] - h2 / 2.0)
        My = max(box1[1] + h1 / 2.0, box2[1] + h2 / 2.0)

    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        iou = 0.0
        if iou_type == 'iou':
            return 0.0
    else:
        area1 = w1 * h1
        area2 = w2 * h2
        carea = cw * ch
        uarea = area1 + area2 - carea
        iou = carea / uarea

    if iou_type == 'iou':
        return iou

    c_area = uw * uh if uw > 0 and uh > 0 else 1e-16

    if iou_type == 'giou':
        giou = iou - (c_area - (area1 + area2 - carea if cw > 0 and ch > 0 else area1 + area2)) / c_area
        return giou

    centre_dist2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    diag_len2 = uw * uw + uh * uh + 1e-16
    diou = iou - centre_dist2 / diag_len2

    if iou_type == 'diou':
        return diou

    if iou_type == 'ciou':
        try:
            v = (4.0 / (math.pi ** 2)) * (math.atan(w1 / (h1 + 1e-16)) - math.atan(w2 / (h2 + 1e-16))) ** 2
        except ZeroDivisionError:
            v = 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha = v / (1.0 - iou + v + 1e-16) if (1.0 - iou + v) > 0 else 0.0
            if iou < 0.5:
                alpha = 0.0
        ciou = diou - alpha * v
        return ciou

    return iou


def bboxes_iou_np(boxes_a, boxes_b, xyxy=True, iou_type='iou'):

    iou_type = validate_iou_type(iou_type)

    if xyxy:
        x1_a, y1_a, x2_a, y2_a = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
        x1_b, y1_b, x2_b, y2_b = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
        w_a, h_a = x2_a - x1_a, y2_a - y1_a
        w_b, h_b = x2_b - x1_b, y2_b - y1_b
        cx_a = (x1_a + x2_a) / 2.0
        cy_a = (y1_a + y2_a) / 2.0
        cx_b = (x1_b + x2_b) / 2.0
        cy_b = (y1_b + y2_b) / 2.0
    else:
        cx_a, cy_a, w_a, h_a = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
        cx_b, cy_b, w_b, h_b = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
        x1_a = cx_a - w_a / 2.0
        y1_a = cy_a - h_a / 2.0
        x2_a = cx_a + w_a / 2.0
        y2_a = cy_a + h_a / 2.0
        x1_b = cx_b - w_b / 2.0
        y1_b = cy_b - h_b / 2.0
        x2_b = cx_b + w_b / 2.0
        y2_b = cy_b + h_b / 2.0

    xx1 = np.maximum(x1_a[:, None], x1_b[None, :])
    yy1 = np.maximum(y1_a[:, None], y1_b[None, :])
    xx2 = np.minimum(x2_a[:, None], x2_b[None, :])
    yy2 = np.minimum(y2_a[:, None], y2_b[None, :])

    w_inter = np.maximum(0.0, xx2 - xx1)
    h_inter = np.maximum(0.0, yy2 - yy1)
    area_inter = w_inter * h_inter

    area_a = w_a * h_a
    area_b = w_b * h_b
    area_union = area_a[:, None] + area_b[None, :] - area_inter
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.where(area_union > 0, area_inter / (area_union + 1e-16), 0.0)

    if iou_type == 'iou':
        return iou

    cx1 = np.minimum(x1_a[:, None], x1_b[None, :])
    cy1 = np.minimum(y1_a[:, None], y1_b[None, :])
    cx2 = np.maximum(x2_a[:, None], x2_b[None, :])
    cy2 = np.maximum(y2_a[:, None], y2_b[None, :])
    cw = np.maximum(0.0, cx2 - cx1)
    ch = np.maximum(0.0, cy2 - cy1)
    area_c = cw * ch + 1e-16

    if iou_type == 'giou':
        giou = iou - (area_c - area_union) / area_c
        return giou

    centre_dist2 = (cx_a[:, None] - cx_b[None, :]) ** 2 + (cy_a[:, None] - cy_b[None, :]) ** 2
    diag_len2 = cw ** 2 + ch ** 2 + 1e-16
    diou = iou - centre_dist2 / diag_len2

    if iou_type == 'diou':
        return diou

    if iou_type == 'ciou':
        with np.errstate(divide='ignore', invalid='ignore'):
            arctan_a = np.arctan(w_a / (h_a + 1e-16))
            arctan_b = np.arctan(w_b / (h_b + 1e-16))
            v = (4.0 / (math.pi ** 2)) * (arctan_a[:, None] - arctan_b[None, :]) ** 2
            denom = 1.0 - iou + v + 1e-16
            alpha_raw = v / denom
            alpha = np.where(iou >= 0.5, alpha_raw, 0.0)
        ciou = diou - alpha * v
        return ciou

    return iou


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False, iou_type='iou'):

    iou_type = validate_iou_type(iou_type)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        if idx_other.size == 0:
            break

        if iou_type == 'iou' and not min_mode:
            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            over = inter / (areas[order[0]] + areas[order[1:]] - inter)
        else:
            boxes_self = boxes[idx_self:idx_self + 1]
            boxes_other = boxes[idx_other]
            if min_mode:
                iou_mat = bboxes_iou_np(boxes_self, boxes_other, xyxy=True, iou_type='iou')
                min_areas = np.minimum(areas[idx_self], areas[idx_other])
                raw_over = iou_mat[0]
                over = np.where(min_areas > 0, (raw_over * (areas[idx_self] + areas[idx_other] - (raw_over * min_areas if False else 0))) / min_areas, 0.0)
                xx1 = np.maximum(x1[idx_self], x1[idx_other])
                yy1 = np.maximum(y1[idx_self], y1[idx_other])
                xx2 = np.minimum(x2[idx_self], x2[idx_other])
                yy2 = np.minimum(y2[idx_self], y2[idx_other])
                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h
                over = np.where(min_areas > 0, inter / min_areas, 0.0)
            else:
                iou_mat = bboxes_iou_np(boxes_self, boxes_other, xyxy=True, iou_type=iou_type)
                over = iou_mat[0]

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)



def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        bbox_thick = int(0.6 * (height + width) / 600)
        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue = get_color(0, offset, classes)
            if color is None:
                rgb = (red, green, blue)
            msg = str(class_names[cls_id])+" "+str(round(cls_conf,3))
            t_size = cv2.getTextSize(msg, 0, 0.7, thickness=bbox_thick // 2)[0]
            c1, c2 = (x1,y1), (x2, y2)
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(img, (x1,y1), (np.float32(c3[0]), np.float32(c3[1])), rgb, -1)
            img = cv2.putText(img, msg, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,0), bbox_thick//2,lineType=cv2.LINE_AA)
        
        img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, bbox_thick)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size / 5, 5)  # to avoid single truth problem
        return truths
    else:
        return np.array([])


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names



def post_processing(img, conf_thresh, nms_thresh, output, iou_type='iou'):

    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]

    t1 = time.time()

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    t2 = time.time()

    bboxes_batch = []
    for i in range(box_array.shape[0]):
       
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh, iou_type=iou_type)
            
            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])
        
        bboxes_batch.append(bboxes)

    t3 = time.time()

    print('-----------------------------------')
    print('       max and argmax : %f' % (t2 - t1))
    print('                  nms : %f' % (t3 - t2))
    print('Post processing total : %f' % (t3 - t1))
    print('-----------------------------------')
    
    return bboxes_batch
