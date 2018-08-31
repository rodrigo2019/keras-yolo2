# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:44:54 2018

@author: Rodrigo.Andrade
"""
from keras.models import Model, load_model
from keras.layers import  Input, MaxPooling2D, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger
from keras.utils import Sequence
from imgaug import augmenters as iaa
from backend import BaseFeatureExtractor
from utils import list_images, import_feature_extractor, get_session, create_backup
import cv2
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

class BatchGenerator(Sequence):
    def __init__(self, images_paths, 
                       config, 
                       shuffle=True, 
                       jitter=True, 
                       norm=None):
        self.generator = None

        self.config = config

        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm

        self.images = []
        self.labels = {}
        for fname in images_paths:
            image = {}
            image['filename'] = fname
            obj_class=os.path.normpath(fname)
            obj_class=obj_class.split(os.sep)[-2]
            if obj_class not in self.labels:
                self.labels[obj_class] = len(self.labels)
            image['class'] = self.labels[obj_class]
            self.images.append(image)

        ### augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5), # horizontally flip 50% of all images
                iaa.Flipud(0.2), # vertically flip 20% of all images
                #sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-5, 5), # rotate by -45 to +45 degrees
                    shear=(-5, 5), # shear by -16 to +16 degrees
                    #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges
                        #sometimes(iaa.OneOf([
                        #    iaa.EdgeDetect(alpha=(0, 0.7)),
                        #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                        #])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                            #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        #iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                        #iaa.Grayscale(alpha=(0.0, 1.0)),
                        #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )
        if shuffle: np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))   

    def num_classes(self):
        return len(self.labels)

    def size(self):
        return len(self.images)

    def load_image(self, i):
        if self.config['IMAGE_C'] == 1:
            image = cv2.imread(self.images[i]['filename'], cv2.IMREAD_GRAYSCALE)
            image = image[:,:,np.newaxis]
        elif self.config['IMAGE_C'] == 3:
            image = cv2.imread(self.images[i]['filename'])
        else:
            raise ValueError("Invalid number of image channels.")
        return image


    def __getitem__(self, idx):
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0
        if self.config['IMAGE_C'] == 3:
            x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))                         # input images
        else:
            x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 1))
        if len(self.labels)>2:
            y_batch = np.zeros((r_bound - l_bound, len(self.labels)))
        else:
            y_batch = np.zeros((r_bound - l_bound, 1))

        for train_instance in self.images[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img = self.aug_image(train_instance, jitter=self.jitter)
            img = cv2.resize(img,(self.config['IMAGE_W'], self.config['IMAGE_H']))       
            # assign input image to x_batch
            if self.norm != None: 
                x_batch[instance_count] = self.norm(img)
            else:
                x_batch[instance_count] = img

            if len(self.labels)>2:
                y = np.zeros(len(self.labels))
                y[train_instance['class']] = 1.0    
            else:
                y = np.zeros(1)
                y[0] = train_instance['class']
            y_batch[instance_count] = y      
            

            # increase instance counter in current batch
            instance_count += 1  

        #print(' new batch created', idx)

        return x_batch, y_batch


    def aug_image(self, train_instance, jitter):

        image_name = train_instance['filename']
        if self.config['IMAGE_C'] == 1:
            image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        elif self.config['IMAGE_C'] == 3:
            image = cv2.imread(image_name)
        else:
            raise ValueError("Invalid number of image channels.")

        if image is None: print('Cannot find ', image_name)
    
        if jitter:   
            image = self.aug_pipe.augment_image(image) 
            
        return image


def _main_(args):

    config_path = args.conf
    
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    if config['backup']['create_backup']:
        config = create_backup(config)

    keras.backend.tensorflow_backend.set_session(get_session())

    #path for the training and validation dataset
    datasetTrainPath = os.path.join(args.folder,"train")
    datasetValPath = os.path.join(args.folder,"val")

    for folder in [datasetTrainPath, datasetValPath]:
        if not os.path.isdir(folder):
            raise Exception("{} doesn't exist!".format(folder))

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
    
    generator_config = {
        'IMAGE_H'         : height, 
        'IMAGE_W'         : width,
        'IMAGE_C'         : depth,
        'BATCH_SIZE'      : batchSize
    }  

    #callbacks    
    model_name = config['train']['saved_weights_name']
    checkPointSaverBest=ModelCheckpoint(model_name, monitor='val_acc', verbose=1, 
                                        save_best_only=True, save_weights_only=False, mode='auto', period=1)
    ckp_model_name = os.path.splitext(model_name)[1]+"_ckp.h5"
    checkPointSaver=ModelCheckpoint(ckp_model_name, verbose=1, 
                                save_best_only=False, save_weights_only=False, period=10)

    tb=TensorBoard(log_dir=config['train']['tensorboard_log_dir'], histogram_freq=0, batch_size=batchSize, write_graph=True,
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

    train_generator = BatchGenerator(imagesTrainPaths, 
                                generator_config, 
                                norm=feature_extractor.normalize,
                                jitter=True)
    val_generator = BatchGenerator(imagesValPaths, 
                                    generator_config, 
                                    norm=feature_extractor.normalize,
                                    jitter=False)  

    features = feature_extractor.extract(input_image)          

    # make the model head
    output = Conv2D(classes, (1, 1), padding="same")(features)
    output = BatchNormalization()(output)
    output = LeakyReLU(alpha=0.1)(output)
    output = GlobalAveragePooling2D()(output)
    output = Activation("sigmoid")(output) if classes == 1 else Activation("softmax")(output)

    if config['model']['pretrained_weights'] != "":
        model = load_model(config['model']['pretrained_weights'] )
    else:
        model = Model(input_image, output)   
        opt = Adam()
        model.compile(loss="binary_crossentropy" if classes == 1 else "categorical_crossentropy",
                    optimizer=opt,metrics=["accuracy"])
    model.summary()

    model.fit_generator(
            train_generator,
            steps_per_epoch=len(imagesTrainPaths)//batchSize,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=len(imagesValPaths)//batchSize,
            callbacks=[checkPointSaverBest,checkPointSaver,tb],
            workers=12,
            max_queue_size=40)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)