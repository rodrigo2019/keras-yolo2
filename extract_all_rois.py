
from preprocessing import parse_annotation_xml, parse_annotation_csv
from tqdm import tqdm
import numpy as np
import json
import argparse
import keras
import os
import cv2

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default='config.json',
    help='path to configuration file')

def _main_(args):
    config_path = args.conf
    
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

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
        print('No labels are provided. Saving on all seen labels.')
        config['model']['labels'] = train_labels.keys()
        with open("labels.json", 'w') as outfile:
            json.dump({"labels" : list(train_labels.keys())},outfile)

    if os.path.isdir("./roi_dataset"):
        print("roi_dataset already exists, please move or delete it first.")
        return
    else:
        os.mkdir("roi_dataset")
        os.mkdir("roi_dataset/train")
        os.mkdir("roi_dataset/val")
        
        all_imgs = [train_imgs, valid_imgs]
        for j,folder_name in enumerate(["train", "val"]):
            print("generating", folder_name)
            for img in tqdm(all_imgs[j]):
                image = cv2.imread(img['filename'])
                for i,obj in enumerate(img['object']):
                    xmin = obj['xmin']
                    ymin = obj['ymin']
                    xmax = obj['xmax']
                    ymax = obj['ymax']
                    name = obj['name']
                    if not os.path.isdir("roi_dataset/{}/{}".format(folder_name, name)): 
                        os.mkdir("roi_dataset/{}/{}".format(folder_name, name))
                    roi = image[ymin:ymax, xmin:xmax]
                    base_name = os.path.basename(img['filename'])
                    base_name, ext = os.path.splitext(base_name)
                    cv2.imwrite("roi_dataset/{}/{}/{}_{}_{}.jpg".format(folder_name,name,name, base_name, i), roi)

if __name__ == "__main__":
    args = argparser.parse_args()
    _main_(args)