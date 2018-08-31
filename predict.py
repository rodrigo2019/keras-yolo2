#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes, get_session
from frontend import YOLO
from utils import list_images
import keras
import json
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    keras.backend.tensorflow_backend.set_session(get_session())

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    if weights_path == '':
        weights_path = config['train']['saved_weights_name']

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = (config['model']['input_size_h'],config['model']['input_size_w']), 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'],
                gray_mode           = config['model']['gray_mode'])

    ###############################
    #   Load trained weights
    ###############################    

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    if image_path[-4:] == '.mp4':
        video_out = image_path[:-4] + '_detected' + image_path[-4:]
        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'), 
                               50.0, 
                               (frame_w, frame_h))

        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()
            
            boxes = yolo.predict(image)
            image = draw_boxes(image, boxes, config['model']['labels'])

            video_writer.write(np.uint8(image))

        video_reader.release()
        video_writer.release()  
    else:
        if os.path.isfile(image_path):
            image = cv2.imread(image_path)
            boxes = yolo.predict(image)
            image = draw_boxes(image, boxes, config['model']['labels'])

            print(len(boxes), 'boxes are found')

            cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)
        else:
            detected_images_path = os.path.join(image_path, "detected")
            if not os.path.exists(detected_images_path):
                os.mkdir(detected_images_path)
            images = list(list_images(image_path))
            for fname in tqdm(images):
                image = cv2.imread(fname)
                boxes = yolo.predict(image)
                image = draw_boxes(image, boxes, config['model']['labels'])
                fname = os.path.basename(fname)
                cv2.imwrite(os.path.join(image_path, "detected", fname), image)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
