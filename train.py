#! /usr/bin/env python3

from keras_yolov2.preprocessing import parse_annotation_xml, parse_annotation_csv
from keras_yolov2.utils import get_session, create_backup
from keras_yolov2.frontend import YOLO
import numpy as np
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


def _main_(args):
    config_path = args.conf

    keras.backend.tensorflow_backend.set_session(get_session())

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    if config['backup']['create_backup']:
        config = create_backup(config)
    ###############################
    #   Parse the annotations 
    ###############################

    if config['parser_annotation_type'] == 'xml':
        # parse annotations of the training set
        train_imgs, train_labels = parse_annotation_xml(config['train']['train_annot_folder'],
                                                        config['train']['train_image_folder'],
                                                        config['model']['labels'])

        # parse annotations of the validation set, if any, otherwise split the training set
        if os.path.exists(config['valid']['valid_annot_folder']):
            valid_imgs, valid_labels = parse_annotation_xml(config['valid']['valid_annot_folder'],
                                                            config['valid']['valid_image_folder'],
                                                            config['model']['labels'])
            split = False
        else:
            split = True
    elif config['parser_annotation_type'] == 'csv':
        # parse annotations of the training set
        train_imgs, train_labels = parse_annotation_csv(config['train']['train_csv_file'],
                                                        config['model']['labels'],
                                                        config['train']['train_csv_base_path'])

        # parse annotations of the validation set, if any, otherwise split the training set
        if os.path.exists(config['valid']['valid_csv_file']):
            valid_imgs, valid_labels = parse_annotation_csv(config['valid']['valid_csv_file'],
                                                            config['model']['labels'],
                                                            config['valid']['valid_csv_base_path'])
            split = False
        else:
            split = True
    else:
        raise ValueError("'parser_annotations_type' must be 'xml' or 'csv' not {}.".format(config['parser_annotations_type']))

    if split:
        train_valid_split = int(0.8*len(train_imgs))
        np.random.shuffle(train_imgs)

        valid_imgs = train_imgs[train_valid_split:]
        train_imgs = train_imgs[:train_valid_split]

    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

        print('Seen labels:\t', train_labels)
        print('Given labels:\t', config['model']['labels'])
        print('Overlap labels:\t', overlap_labels)

        if len(overlap_labels) < len(config['model']['labels']):
            print('Some labels have no annotations! Please revise the list of labels in the config.json file!')
            return
    else:
        print('No labels are provided. Train on all seen labels.')
        config['model']['labels'] = train_labels.keys()
        with open("labels.json", 'w') as outfile:
            json.dump({"labels": list(train_labels.keys())}, outfile)

    ###############################
    #   Construct the model 
    ###############################

    yolo = YOLO(backend=config['model']['backend'],
                input_size=(config['model']['input_size_h'], config['model']['input_size_w']),
                labels=config['model']['labels'],
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'],
                gray_mode=config['model']['gray_mode'])

    #########################################
    #   Load the pretrained weights (if any) 
    #########################################

    if os.path.exists(config['train']['pretrained_weights']):
        print("Loading pre-trained weights in", config['train']['pretrained_weights'])
        yolo.load_weights(config['train']['pretrained_weights'])

    ###############################
    #   Start the training process 
    ###############################

    yolo.train(train_imgs=train_imgs,
               valid_imgs=valid_imgs,
               train_times=config['train']['train_times'],
               valid_times=config['valid']['valid_times'],
               nb_epochs=config['train']['nb_epochs'],
               learning_rate=config['train']['learning_rate'],
               batch_size=config['train']['batch_size'],
               warmup_epochs=config['train']['warmup_epochs'],
               object_scale=config['train']['object_scale'],
               no_object_scale=config['train']['no_object_scale'],
               coord_scale=config['train']['coord_scale'],
               class_scale=config['train']['class_scale'],
               saved_weights_name=config['train']['saved_weights_name'],
               debug=config['train']['debug'],
               early_stop=config['train']['early_stop'],
               workers=config['train']['workers'],
               max_queue_size=config['train']['max_queue_size'],
               tb_logdir=config['train']['tensorboard_log_dir'],
               train_generator_callback=config['train']['callback'],
               iou_threshold=config['valid']['iou_threshold'],
               score_threshold=config['valid']['score_threshold'])


if __name__ == '__main__':
    _args = argparser.parse_args()
    _main_(_args)
