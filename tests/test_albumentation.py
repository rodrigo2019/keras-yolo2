import sys

sys.path.append("..")
from keras_yolov2.preprocessing import parse_annotation_csv
import cv2
import time

from albumentations import (
    CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
import numpy as np


def strong_aug(p=0.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)

augmentation = strong_aug(p=0.9)

samples = parse_annotation_csv("/home/rodrigo/Documents/freeflow_dataset/cheetah_anita/train_allCar.txt",
                               [],
                               "/home/rodrigo/Documents/freeflow_dataset/cheetah_anita/train")[0]

start_time = time.time()
for sample in samples:
    image = cv2.imread(sample["filename"])
    data = {"image": image}
    augmented = augmentation(**data)
    image = augmented["image"]
    cv2.imshow("sample", image)
    # cv2.imshow("orig", image_orig)
    if cv2.waitKey(1) == 27:
        break
print(time.time() - start_time)
