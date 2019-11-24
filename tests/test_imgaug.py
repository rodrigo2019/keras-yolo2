import sys

sys.path.append("..")
from keras_yolov2.preprocessing import parse_annotation_csv
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import time

# augmentors by https://github.com/aleju/imgaug
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
aug_pipe = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.2),  # vertically flip 20% of all images
        # sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent
            rotate=(-20, 20),  # rotate by -45 to +45 degrees
            shear=(-5, 5),  # shear by -16 to +16 degrees
            # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
                   [
                       # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(2, 7)),
                           # blur image using local means (kernel sizes between 2 and 7)
                           iaa.MedianBlur(k=(3, 11)),
                           # blur image using local medians (kernel sizes between 2 and 7)
                       ]),
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                       iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                       # search either for all edges or for directed edges
                       # sometimes(iaa.OneOf([
                       #    iaa.EdgeDetect(alpha=(0, 0.7)),
                       #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                       # ])),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                       # add gaussian noise
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                           # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                       ]),
                       iaa.Invert(0.05, per_channel=True),  # invert color channels
                       iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images
                       iaa.Multiply((0.5, 1.5), per_channel=0.5),  # change brightness of images
                       iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                   ],
                   random_order=True
                   )
    ],
    random_order=True
)
samples = parse_annotation_csv("/home/rodrigo/Documents/freeflow_dataset/cheetah_anita/train_allCar.txt",
                               [],
                               "/home/rodrigo/Documents/freeflow_dataset/cheetah_anita/train")[0]

start_time = time.time()
for sample in samples:
    image = cv2.imread(sample["filename"])
    # image_orig = image.copy()
    bbs = []
    for obj in sample["object"]:
        xmin = obj["xmin"]
        ymin = obj["ymin"]
        xmax = obj["xmax"]
        ymax = obj["ymax"]
        # cv2.rectangle(image_orig, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
        bbs.append(BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax))
    bbs = BoundingBoxesOnImage(bbs, shape=image.shape)
    image, bbs = aug_pipe(image=image, bounding_boxes=bbs)
    # print("new")
    for bbox in bbs.bounding_boxes:
        xmin = int(bbox.x1)
        xmax = int(bbox.x2)
        ymin = int(bbox.y1)
        ymax = int(bbox.y2)
        # print(xmin, xmax, ymin, ymax)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)

    cv2.imshow("sample", image)
    # cv2.imshow("orig", image_orig)
    if cv2.waitKey(1) == 27:
        break
print(time.time() - start_time)
