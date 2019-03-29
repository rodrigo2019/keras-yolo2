from .yolo_loss import YoloLoss
from .map_evaluation import MapEvaluation
from .utils import decode_netout, import_feature_extractor, import_dynamically
from .preprocessing import BatchGenerator
from keras.models import Model
from keras.layers import Reshape, Conv2D, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import numpy as np
import sys
import cv2
import os


class YOLO(object):
    def __init__(self, backend, input_size, labels, max_box_per_image, anchors, gray_mode=False):

        self._input_size = input_size
        self._gray_mode = gray_mode
        self.labels = list(labels)
        self._nb_class = len(self.labels)
        self._nb_box = len(anchors) // 2
        self._anchors = anchors

        self._max_box_per_image = max_box_per_image

        ##########################
        # Make the model
        ##########################

        # make the feature extractor layers
        if self._gray_mode:
            self._input_size = (self._input_size[0], self._input_size[1], 1)
            input_image = Input(shape=self._input_size)
        else:
            self._input_size = (self._input_size[0], self._input_size[1], 3)
            input_image = Input(shape=self._input_size)

        self._feature_extractor = import_feature_extractor(backend, self._input_size)

        print(self._feature_extractor.get_output_shape())
        self._grid_h, self._grid_w = self._feature_extractor.get_output_shape()
        features = self._feature_extractor.extract(input_image)

        # make the object detection layer
        output = Conv2D(self._nb_box * (4 + 1 + self._nb_class),
                        (1, 1), strides=(1, 1),
                        padding='same',
                        name='Detection_layer',
                        kernel_initializer='lecun_normal')(features)
        output = Reshape((self._grid_h, self._grid_w, self._nb_box, 4 + 1 + self._nb_class), name="YOLO_output")(output)

        self._model = Model(input_image, output)

        # initialize the weights of the detection layer
        layer = self._model.get_layer("Detection_layer")
        weights = layer.get_weights()

        new_kernel = np.random.normal(size=weights[0].shape) / (self._grid_h * self._grid_w)
        new_bias = np.random.normal(size=weights[1].shape) / (self._grid_h * self._grid_w)

        layer.set_weights([new_kernel, new_bias])

        # print a summary of the whole model
        self._model.summary()

        # declare class variables
        self._batch_size = None
        self._object_scale = None
        self._no_object_scale = None
        self._coord_scale = None
        self._class_scale = None
        self._debug = None
        self._warmup_batches = None

    def load_weights(self, weight_path):
        self._model.load_weights(weight_path)

    def train(self, train_imgs,  # the list of images to train the model
              valid_imgs,  # the list of images used to validate the model
              train_times,  # the number of time to repeat the training set, often used for small datasets
              valid_times,  # the number of times to repeat the validation set, often used for small datasets
              nb_epochs,  # number of epoches
              learning_rate,  # the learning rate
              batch_size,  # the size of the batch
              warmup_epochs,  # number of initial batches to let the model familiarize with the new dataset
              object_scale,
              no_object_scale,
              coord_scale,
              class_scale,
              saved_weights_name='best_weights.h5',
              debug=False,
              workers=3,
              max_queue_size=8,
              early_stop=True,
              custom_callback=[],
              tb_logdir="./",
              train_generator_callback=None,
              iou_threshold=0.5,
              score_threshold=0.5):

        self._batch_size = batch_size

        self._object_scale = object_scale
        self._no_object_scale = no_object_scale
        self._coord_scale = coord_scale
        self._class_scale = class_scale

        self._debug = debug

        #######################################
        # Make train and validation generators
        #######################################

        generator_config = {
            'IMAGE_H': self._input_size[0],
            'IMAGE_W': self._input_size[1],
            'IMAGE_C': self._input_size[2],
            'GRID_H': self._grid_h,
            'GRID_W': self._grid_w,
            'BOX': self._nb_box,
            'LABELS': self.labels,
            'CLASS': len(self.labels),
            'ANCHORS': self._anchors,
            'BATCH_SIZE': self._batch_size,
            'TRUE_BOX_BUFFER': self._max_box_per_image,
        }

        if train_generator_callback is not None:
            basepath = os.path.dirname(train_generator_callback)
            sys.path.append(basepath)
            custom_callback_name = os.path.basename(train_generator_callback)
            custom_generator_callback = import_dynamically(custom_callback_name)
        else:
            custom_generator_callback = None

        train_generator = BatchGenerator(train_imgs,
                                         generator_config,
                                         norm=self._feature_extractor.normalize,
                                         callback=custom_generator_callback)
        valid_generator = BatchGenerator(valid_imgs,
                                         generator_config,
                                         norm=self._feature_extractor.normalize,
                                         jitter=False)

        # TODO: warmup is not working with new loss function formula
        self._warmup_batches = warmup_epochs * (train_times * len(train_generator) + valid_times * len(valid_generator))

        ############################################
        # Compile the model
        ############################################

        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        loss_yolo = YoloLoss(self._anchors, (self._grid_w, self._grid_h), self._batch_size,
                             lambda_coord=coord_scale, lambda_noobj=no_object_scale, lambda_obj=object_scale,
                             lambda_class=class_scale)
        self._model.compile(loss=loss_yolo, optimizer=optimizer)

        ############################################
        # Make a few callbacks
        ############################################

        early_stop_cb = EarlyStopping(monitor='val_loss',
                                      min_delta=0.001,
                                      patience=3,
                                      mode='min',
                                      verbose=1)

        tensorboard_cb = TensorBoard(log_dir=tb_logdir,
                                     histogram_freq=0,
                                     # write_batch_performance=True,
                                     write_graph=True,
                                     write_images=False)

        root, ext = os.path.splitext(saved_weights_name)
        ckp_best_loss = ModelCheckpoint(root + "_bestLoss" + ext,
                                        monitor='val_loss',
                                        verbose=1,
                                        save_best_only=True,
                                        mode='min',
                                        period=1)
        ckp_saver = ModelCheckpoint(root + "_ckp" + ext,
                                    verbose=1,
                                    period=10)
        map_evaluator_cb = MapEvaluation(self, valid_generator,
                                         save_best=True,
                                         save_name=root + "_bestMap" + ext,
                                         tensorboard=tensorboard_cb,
                                         iou_threshold=iou_threshold,
                                         score_threshold=score_threshold)

        if not isinstance(custom_callback, list):
            custom_callback = [custom_callback]
        callbacks = [ckp_best_loss, ckp_saver, tensorboard_cb, map_evaluator_cb] + custom_callback
        if early_stop:
            callbacks.append(early_stop_cb)

        #############################
        # Start the training process
        #############################

        self._model.fit_generator(generator=train_generator,
                                  steps_per_epoch=len(train_generator) * train_times,
                                  epochs=warmup_epochs + nb_epochs,
                                  verbose=2 if debug else 1,
                                  validation_data=valid_generator,
                                  validation_steps=len(valid_generator) * valid_times,
                                  callbacks=callbacks,
                                  workers=workers,
                                  max_queue_size=max_queue_size)

    def get_inference_model(self):
        return self._model

    def predict(self, image, iou_threshold=0.5, score_threshold=0.5):

        if len(image.shape) == 3 and self._gray_mode:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = image[..., np.newaxis]
        elif len(image.shape) == 2 and not self._gray_mode:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 2:
            image = image[..., np.newaxis]

        image = cv2.resize(image, (self._input_size[1], self._input_size[0]))
        image = self._feature_extractor.normalize(image)
        if len(image.shape) == 3:
            input_image = image[np.newaxis, :]
        else:
            input_image = image[np.newaxis, ..., np.newaxis]

        netout = self._model.predict(input_image)[0]

        boxes = decode_netout(netout, self._anchors, self._nb_class, score_threshold, iou_threshold)

        return boxes
