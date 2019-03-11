"""
this callback has the purpose to do a very specific job that imgaug can't do easily, for example in this example
the callback changes the background imagem from the original one, using a random picture from kitti dataset
"""
from sys import path
path.append("..")
from keras_yolov2.utils import list_images
import cv2

IMAGES_PATH = "C:\\datasets\\kitti\\data_object_image_2\\training\\image_2"
images = list(list_images(IMAGES_PATH))
counter = 0


def aug_callback(image, instance):
    """
    instance dict structure:
    {
      'object': [
       {'xmin': int, 'xmax': int, 'ymin': int, 'ymax': int, 'name': str},
       .
       .
       .
       {'xmin': int, 'xmax': int, 'ymin': int, 'ymax': int, 'name': str},
      ],
      'filename': str,
      'width': int,
      'height': int
    }
    :param image: [np.array]
    :param instance: [dict]
    :return: tuple([np.array] the new image, [dict] the new dict)
    """
    global counter
    if len(image.shape) == 2:
        new_background = cv2.imread(images[counter], 0)
    else:
        new_background = cv2.imread(images[counter])

    if counter == len(images)-1:
        counter = 0
    else:
        counter += 1

    all_obj = instance["object"]
    for obj in all_obj:
        xmin = obj["xmin"]
        xmax = obj["xmax"]
        ymin = obj["ymin"]
        ymax = obj["ymax"]
        new_background[ymin:ymax, xmin:xmax] = image[ymin:ymax, xmin:xmax]
    instance['width'] = new_background.shape[1]
    instance['height'] = new_background.shape[0]
    # cv2.imshow("im", new_background)
    # cv2.waitKey(0)
    return new_background, instance
