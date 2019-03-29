#! /usr/bin/env python3

from keras_yolov2.utils import get_session
from keras_yolov2.frontend import YOLO
import argparse
import keras
import json
import os

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default='config.json',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    default='',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')


def _main_(args):
    config_path = args.conf
    weights_path = args.weights

    keras.backend.tensorflow_backend.set_session(get_session())

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    if weights_path == '':
        weights_path = config['train']['saved_weights_name']

    yolo = YOLO(backend=config['model']['backend'],
                input_size=(config['model']['input_size_h'], config['model']['input_size_w']),
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'],
                gray_mode=config['model']['gray_mode'])

    yolo.load_weights(weights_path)

    inference_model = yolo.get_inference_model()
    inference_model.save("{}_inference.h5".format(os.path.split(weights_path)[0]))


if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)
