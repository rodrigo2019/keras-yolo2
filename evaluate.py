#! /usr/bin/env python3
from preprocessing import parse_annotation_xml, parse_annotation_csv
from preprocessing import BatchGenerator
from utils import get_session, create_backup
from frontend import YOLO
import numpy as np
import tensorflow as tf
import json
import keras
import argparse
import os

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default='config.json',
    help='path to configuration file')

argparser.add_argument(
    '-i',
    '--iou',
    default=0.5,
    help='IOU threshold',
    type=float)

argparser.add_argument(
    '-w',
    '--weights',
    default='',
    help='path to pretrained weights')

def _main_(args):
    config_path = args.conf
    weights_path = args.weights
    
    keras.backend.tensorflow_backend.set_session(get_session())

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    if weights_path == '':
        weights_path = config['train']['pretrained_weights"']

    ###############################
    #   Parse the annotations 
    ###############################
    without_valid_imgs = False
    if config['parser_annotation_type'] == 'xml':
        # parse annotations of the training set
        train_imgs, train_labels = parse_annotation_xml(config['train']['train_annot_folder'], 
                                                    config['train']['train_image_folder'], 
                                                    config['model']['labels'])

        # parse annotations of the validation set, if any.
        if os.path.exists(config['valid']['valid_annot_folder']):
            valid_imgs, valid_labels = parse_annotation_xml(config['valid']['valid_annot_folder'], 
                                                        config['valid']['valid_image_folder'], 
                                                        config['model']['labels'])
        else:
            without_valid_imgs = True

    elif config['parser_annotation_type'] == 'csv':
        # parse annotations of the training set
        train_imgs, train_labels = parse_annotation_csv(config['train']['train_csv_file'],
                                                        config['model']['labels'],
                                                        config['train']['train_csv_base_path'])

        # parse annotations of the validation set, if any.
        if os.path.exists(config['valid']['valid_csv_file']):
            valid_imgs, valid_labels = parse_annotation_csv(config['valid']['valid_csv_file'],
                                                        config['model']['labels'],
                                                        config['valid']['valid_csv_base_path'])
        else:
            without_valid_imgs = True
    else:
        raise ValueError("'parser_annotations_type' must be 'xml' or 'csv' not {}.".format(config['parser_annotations_type']))

    #remove samples without objects in the image
    for i in range(len(train_imgs)-1,0,-1):
        if len(train_imgs[i]['object']) == 0:
            del train_imgs[i]

    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(train_labels.keys()))

        print('Seen labels:\t', train_labels)
        print('Given labels:\t', config['model']['labels'])
        print('Overlap labels:\t', overlap_labels)           

        if len(overlap_labels) < len(config['model']['labels']):
            print('Some labels have no annotations! Please revise the list of labels in the config.json file!')
            return
    else:
        print('No labels are provided. Evaluate on all seen labels.')
        config['model']['labels'] = train_labels.keys()
        with open("labels.json", 'w') as outfile:
            json.dump({"labels" : list(train_labels.keys())},outfile)
        
    ###############################
    #   Construct the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = (config['model']['input_size_h'], config['model']['input_size_w']), 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'],
                gray_mode           = config['model']['gray_mode'])

    ###############################
    #   Load the pretrained weights (if any) 
    ###############################    

    if weights_path != '':
        print("Loading pre-trained weights in", weights_path)
        yolo.load_weights(weights_path)
    elif os.path.exists(config['train']['pretrained_weights']):
        print("Loading pre-trained weights in", config['train']['pretrained_weights'])
        yolo.load_weights(config['train']['pretrained_weights'])
    else:
        raise Exception("No pretrained weights found.")

    ###############################
    #   Evaluate the network
    ###############################  

    print("calculing mAP for iou threshold = {}".format(args.iou))
    generator_config = {
                'IMAGE_H'         : yolo.input_size[0], 
                'IMAGE_W'         : yolo.input_size[1],
                'IMAGE_C'         : yolo.input_size[2],
                'GRID_H'          : yolo.grid_h,  
                'GRID_W'          : yolo.grid_w,
                'BOX'             : yolo.nb_box,
                'LABELS'          : yolo.labels,
                'CLASS'           : len(yolo.labels),
                'ANCHORS'         : yolo.anchors,
                'BATCH_SIZE'      : 4,
                'TRUE_BOX_BUFFER' : yolo.max_box_per_image,
            } 
    if not without_valid_imgs:
        valid_generator = BatchGenerator(valid_imgs, 
                                     generator_config, 
                                     norm=yolo.feature_extractor.normalize,
                                     jitter=False)  
        valid_eval = YOLO.MAP_evaluation(yolo, valid_generator,
                                    iou_threshold=args.iou)

        mAP, average_precisions = valid_eval.evaluate_mAP()
        for label, average_precision in average_precisions.items():
            print(yolo.labels[label], '{:.4f}'.format(average_precision))
        print('validation dataset mAP: {:.4f}\n'.format(mAP)) 

    train_generator = BatchGenerator(train_imgs, 
                                     generator_config, 
                                     norm=yolo.feature_extractor.normalize,
                                     jitter=False)  
    train_eval = YOLO.MAP_evaluation(yolo, train_generator,
                                iou_threshold=args.iou)

    
    mAP, average_precisions = train_eval.evaluate_mAP()
    for label, average_precision in average_precisions.items():
        print(yolo.labels[label], '{:.4f}'.format(average_precision))
    print('training dataset mAP: {:.4f}'.format(mAP)) 

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
