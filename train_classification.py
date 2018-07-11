# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:44:54 2018

@author: Rodrigo.Andrade
"""
from keras.models import Model
from keras.layers import  Input, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger
from backend import BaseFeatureExtractor
from utils import list_images, import_feature_extractor
import numpy as np
import os
import keras
import json
import argparse

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    default='config.json',
    help='path to configuration file')

argparser.add_argument(
    '-f',
    '--folder',
    default='./roi_dataset',
    help='path to training folder')

def _main_(args):

    config_path = args.conf
    
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    #path for the training and validation dataset
    datasetTrainPath = os.path.join(args.folder,"train")
    datasetValPath = os.path.join(args.folder,"val")

    for folder in [datasetTrainPath, datasetValPath]:
        if not os.path.isdir(folder):
            raise Exception("{} doesn't exist!".format(fodler))

    classesTrain = next(os.walk(datasetTrainPath))[1]
    classesVal = next(os.walk(datasetValPath))[1]

    if not classesVal == classesTrain:
        raise Exception("The training and validation classes must be the same!")
    else:
        folders = classesTrain

    #training configuration
    epochs = config['train']['nb_epochs']
    batchSize = config['train']['batch_size']
    width = config['model']['input_size_w']
    height = config['model']['input_size_h']
    depth = 3 if config['model']['gray_mode'] == False else 1

    
    #config keras generators
    if len(folders) == 2: #if just have 2 classes, the model will have a binary output
        classes = 1
    else:
        classes = len(folders)

    #count all samples
    imagesTrainPaths = []
    imagesValPaths = []
    for folder in folders: 
        imagesTrainPaths+=list(list_images(os.path.join(datasetTrainPath, folder)))
        imagesValPaths+=list(list_images(os.path.join(datasetValPath, folder)))
    
    trainDatagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

    trainGenerator = trainDatagen.flow_from_directory(
            datasetTrainPath,
            target_size=(height, width),
            batch_size=batchSize,
            class_mode="binary" if classes == 1 else "categorical")

    valDatagen = ImageDataGenerator(rescale=1./255)

    valGenerator = valDatagen.flow_from_directory(
            datasetValPath,
            target_size=(height, width),
            batch_size=batchSize,
            class_mode="binary" if classes == 1 else "categorical")

    #callbacks    
    checkPointSaverBest=ModelCheckpoint(config['train']['saved_weights_name'], monitor='val_acc', verbose=1, 
                                        save_best_only=True, save_weights_only=False, mode='auto', period=1)
    checkPointSaver=ModelCheckpoint(config['train']['saved_weights_name']+"_ckp.hdf5", verbose=1, 
                                save_best_only=False, save_weights_only=False, period=10)

    tb=TensorBoard(log_dir='/logsTB', histogram_freq=0, batch_size=batchSize, write_graph=True,
                write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)


    #create the classification model
    # make the feature extractor layers
    if depth == 1:
        input_size = (height, width, 1)
        input_image     = Input(shape=input_size)
    else:
        input_size = (height, width, 3)
        input_image     = Input(shape=input_size)

    feature_extractor = import_feature_extractor(config['model']['backend'], input_size)           
    features = feature_extractor.extract(input_image)          

    # make the model head
    output = Flatten()(features)
    output = Dense(4096, activation="relu")(output)
    output = Dense(2048, activation="relu")(output)
    output = Dense(classes, activation="sigmoid")(output) if classes == 1 else Dense(classes, activation="softmax")(output)

    model = Model(input_image, output)   
    opt = Adadelta()
    model.compile(loss="binary_crossentropy" if classes == 1 else "categorical_crossentropy",
                optimizer=opt,metrics=["accuracy"])
    model.summary()

    model.fit_generator(
            trainGenerator,
            steps_per_epoch=len(imagesTrainPaths)//batchSize,
            epochs=epochs,
            validation_data=valGenerator,
            validation_steps=len(imagesValPaths)//batchSize,
            #callbacks=[checkPointSaverBest,checkPointSaver,tb],
            workers=12,
            max_queue_size=40)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)