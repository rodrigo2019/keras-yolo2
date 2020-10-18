import os
import shutil
import sys
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf

from .backend import (TinyYoloFeature, FullYoloFeature, MobileNetFeature, SqueezeNetFeature, Inception3Feature,
                      VGG16Feature, ResNet50Feature, BaseFeatureExtractor)


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score

    def __repr__(self):
        """
        Helper method for printing the object's values
        :return:
        """
        return "<BoundBox({}, {}, {}, {}, {}, {})>\n".format(
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            self.get_label(),
            self.get_score()
        )


class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def draw_boxes(image, boxes, labels):
    image_h, image_w, _ = image.shape

    color_levels = [0, 255, 128, 64, 32]
    colors = []
    for r in color_levels:
        for g in color_levels:
            for b in color_levels:
                if r == g and r == b:  # prevent grayscale colors
                    continue
                colors.append((b, g, r))

    for box in boxes:
        xmin = int(box.xmin * image_w)
        ymin = int(box.ymin * image_h)
        xmax = int(box.xmax * image_w)
        ymax = int(box.ymax * image_h)

        line_width_factor = int(min(image_h, image_w) * 0.005)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[box.get_label()], line_width_factor * 2)
        cv2.putText(image, "{} {:.3f}".format(labels[box.get_label()], box.get_score()),
                    (xmin, ymin - line_width_factor * 3), cv2.FONT_HERSHEY_PLAIN, 2e-3 * min(image_h, image_w),
                    (0, 255, 0), line_width_factor)

    return image


def decode_netout(netout, anchors, nb_class, obj_threshold=0.5, nms_threshold=0.3):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []

    # decode the output by the network
    netout[..., 4] = _sigmoid(netout[..., 4])
    netout[..., 5:] = _softmax(netout[..., 5:])

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row, col, b, 5:]
                confidence = netout[row, col, b, 4]

                if confidence >= obj_threshold:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row, col, b, :4]

                    x = (col + _sigmoid(x)) / grid_w  # center position, unit: image width
                    y = (row + _sigmoid(y)) / grid_h  # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w  # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h  # unit: image height

                    box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, confidence, classes)
                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_i].classes[c] = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]

    return boxes


def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x / np.min(x) * t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)


def import_dynamically(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def import_feature_extractor(backend, input_size):
    if backend == 'Inception3':
        feature_extractor = Inception3Feature(input_size)
    elif backend == 'SqueezeNet':
        feature_extractor = SqueezeNetFeature(input_size)
    elif backend == 'MobileNet':
        feature_extractor = MobileNetFeature(input_size)
    elif backend == 'Full Yolo':
        feature_extractor = FullYoloFeature(input_size)
    elif backend == 'Tiny Yolo':
        feature_extractor = TinyYoloFeature(input_size)
    elif backend == 'VGG16':
        feature_extractor = VGG16Feature(input_size)
    elif backend == 'ResNet50':
        feature_extractor = ResNet50Feature(input_size)
    elif os.path.dirname(backend) != "":
        base_path = os.path.dirname(backend)
        sys.path.append(base_path)
        custom_backend_name = os.path.basename(backend)
        custom_backend = import_dynamically(custom_backend_name)
        feature_extractor = custom_backend(input_size)
        if not issubclass(custom_backend, BaseFeatureExtractor):
            raise RuntimeError('You are trying to import a custom backend, your backend must be in inherited from '
                               ' "backend.BaseFeatureExtractor".')
        print('Using a custom backend called {}.'.format(custom_backend_name))
    else:
        raise RuntimeError('Architecture not supported! Only support Full Yolo, Tiny Yolo, MobileNet, SqueezeNet, VGG16'
                           ', ResNet50, or Inception3 at the moment!')

    return feature_extractor


# these funcition are from imutils, you can check this library here: https://github.com/jrosebr1/imutils
# just added this function to have less dependencies
def list_images(base_path, valid_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"), contains=None):
    # return the set of files that are valid
    return list_files(base_path, valid_exts, contains=contains)


def list_files(base_path, valid_exts="", contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(base_path):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(valid_exts):
                # construct the path to the image and yield it
                image_path = os.path.join(rootDir, filename)
                yield image_path


def enable_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def create_backup(config):
    # TODO: the backup creation should be improved, because if the keras_yolo is used as module and the train.py and
    # config.json is in another folder, it wont be copied to the backup

    backup_folder = config['backup']['backup_path']
    prefix = config['backup']['backup_prefix']
    backup_id = datetime.now().strftime('%Y%m%d%H%M%S')
    train_folder_name = "_".join([prefix, backup_id])
    path = os.path.join(backup_folder, train_folder_name)
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)

    shutil.copytree(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."), os.path.join(path, "Keras-yolo2"),
                    ignore=shutil.ignore_patterns(".git"))
    readme_message = ""
    while readme_message == "":
        readme_message = input("Insert a comment about this training: ")
    with open(os.path.join(path, "readme.txt"), 'w') as readme_file:
        readme_file.write(readme_message)

    if config['backup']['redirect_model']:
        model_name = ".".join([train_folder_name, "h5"])
        model_name = os.path.join(path, model_name)
        log_name = os.path.join(path, "logs")
        print('\n\nRedirecting {} file name to {}.'.format(config['train']['saved_weights_name'], model_name))
        print('Redirecting {} tensorborad log to {}.'.format(config['train']['tensorboard_log_dir'], log_name))
        config['train']['saved_weights_name'] = model_name
        config['train']['tensorboard_log_dir'] = log_name

    return config
