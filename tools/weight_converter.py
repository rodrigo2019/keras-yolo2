import sys

sys.path.append("..")

from keras_yolov2.frontend import YOLO
from keras_yolov2.utils import WeightReader
import numpy as np

BACKEND = "Full Yolo"
INPUT_SIZE = (416, 416)
YOLO_ORIGINAL_WEIGHTS = "./yolov2.weights"
yolo = YOLO(backend="Full Yolo",
            input_size=INPUT_SIZE,
            labels=[""]*80,
            anchors=[0] * 10,
            gray_mode=False)

weight_reader = WeightReader(YOLO_ORIGINAL_WEIGHTS)
weight_reader.reset()
nb_conv = 22

for i in range(1, nb_conv + 1):
    conv_layer = yolo.model.layers[1].get_layer('conv_{}'.format(i))
    norm_layer = yolo.model.layers[1].get_layer('norm_{}'.format(i))
    size = np.prod(norm_layer.get_weights()[0].shape)

    beta = weight_reader.read_bytes(size)
    gamma = weight_reader.read_bytes(size)
    mean = weight_reader.read_bytes(size)
    var = weight_reader.read_bytes(size)

    weights = norm_layer.set_weights([gamma, beta, mean, var])

    kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
    kernel = kernel.transpose([2, 3, 1, 0])
    conv_layer.set_weights([kernel])


conv_layer = yolo.model.get_layer('Detection_layer')
bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
kernel = kernel.transpose([2, 3, 1, 0])
conv_layer.set_weights([kernel, bias])

yolo.model.save("teste.h5")
