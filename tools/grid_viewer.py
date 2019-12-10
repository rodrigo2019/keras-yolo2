import sys

sys.path.append("..")
from keras_yolov2.preprocessing import parse_annotation_xml, parse_annotation_csv
from keras_yolov2.utils import import_feature_extractor
import argparse
import json
import cv2

argparser = argparse.ArgumentParser()

argparser.add_argument(
    '-c',
    '--conf',
    default='../config.json',
    help='path to configuration file')


def main(args):
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    if config['parser_annotation_type'] == 'xml':
        # parse annotations of the training set
        train_imgs, train_labels = parse_annotation_xml(config['train']['train_annot_folder'],
                                                        config['train']['train_image_folder'],
                                                        config['model']['labels'])
    elif config['parser_annotation_type'] == 'csv':
        # parse annotations of the training set
        train_imgs, train_labels = parse_annotation_csv(config['train']['train_csv_file'],
                                                        config['model']['labels'],
                                                        config['train']['train_csv_base_path'])

    input_size = (config['model']['input_size_h'], config['model']['input_size_w'], 3)
    feature_extractor = import_feature_extractor(config['model']['backend'], input_size)
    grid_cell_w = round(config['model']['input_size_w'] / feature_extractor.get_output_shape()[1])
    grid_cell_h = round(config['model']['input_size_h'] / feature_extractor.get_output_shape()[0])

    image_w = int(config['model']['input_size_w'])
    image_h = int(config['model']['input_size_h'])

    print("grid cell width:", grid_cell_w)
    print("grid cell height:", grid_cell_h)
    print("grid size:", feature_extractor.get_output_shape())
    for train_img in train_imgs:
        image = cv2.imread(train_img["filename"])
        image = cv2.resize(image, (image_w, image_h))
        for i in range(0, image_w, grid_cell_w):
            cv2.line(image, (i, 0), (i, image_h), (0, 0, 255), 2)
        for i in range(0, image_h, grid_cell_h):
            cv2.line(image, (0, i), (image_w, i), (0, 0, 255), 2)

        cv2.imshow("grid viewer", image)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break


if __name__ == "__main__":
    _args = argparser.parse_args()
    main(_args)
