import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

EPSILON = 1e-7


def calculate_ious(a1, a2, use_iou=True):
    if not use_iou:
        return a1[..., 4]

    def process_boxes(a):
        # ALign x-w, y-h
        a_xy = a[..., 0:2]
        a_wh = a[..., 2:4]

        a_wh_half = a_wh / 2.
        # Get x_min, y_min
        a_mins = a_xy - a_wh_half
        # Get x_max, y_max
        a_maxes = a_xy + a_wh_half

        return a_mins, a_maxes, a_wh

    # Process two sets
    a2_mins, a2_maxes, a2_wh = process_boxes(a2)
    a1_mins, a1_maxes, a1_wh = process_boxes(a1)

    # Intersection as min(Upper1, Upper2) - max(Lower1, Lower2)
    intersect_mins = K.maximum(a2_mins, a1_mins)
    intersect_maxes = K.minimum(a2_maxes, a1_maxes)

    # Getting the intersections in the xy (aka the width, height intersection)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)

    # Multiply to get intersecting area
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    # Values for the single sets
    true_areas = a1_wh[..., 0] * a1_wh[..., 1]
    pred_areas = a2_wh[..., 0] * a2_wh[..., 1]

    # Compute union for the IoU
    union_areas = pred_areas + true_areas - intersect_areas
    return intersect_areas / union_areas


class YoloLoss(object):

    def __init__(self, anchors, grid_size, batch_size, lambda_coord=5, lambda_noobj=1, lambda_obj=1, lambda_class=1,
                 iou_filter=0.6):
        self.__name__ = 'yolo_loss'
        self.iou_filter = iou_filter
        self.readjust_obj_score = False

        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_class = lambda_class

        self.batch_size = batch_size
        self.grid_size = grid_size
        self.nb_anchors = len(anchors) // 2
        self.anchors = np.reshape(anchors, [1, 1, 1, self.nb_anchors, 2])

        self.c_grid = self._generate_yolo_grid(self.batch_size, self.grid_size, self.nb_anchors)

    @staticmethod
    def _generate_yolo_grid(batch_size, grid_size, nb_box):
        cell_x = tf.cast(tf.reshape(tf.tile(tf.range(grid_size[0]), [grid_size[1]]), (1, grid_size[1], grid_size[0],
                                                                                      1, 1)), tf.float32)
        cell_y = tf.cast(tf.reshape(tf.tile(tf.range(grid_size[1]), [grid_size[0]]), (1, grid_size[0], grid_size[1],
                                                                                      1, 1)), tf.float32)
        cell_y = tf.transpose(cell_y, (0, 2, 1, 3, 4))

        cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [batch_size, 1, 1, nb_box, 1])
        return cell_grid

    def _transform_netout(self, y_pred_raw):
        y_pred_xy = K.sigmoid(y_pred_raw[..., :2]) + self.c_grid
        y_pred_wh = K.exp(y_pred_raw[..., 2:4]) * self.anchors
        y_pred_conf = K.sigmoid(y_pred_raw[..., 4:5])
        y_pred_class = y_pred_raw[..., 5:]

        return K.concatenate([y_pred_xy, y_pred_wh, y_pred_conf, y_pred_class], axis=-1)

    def coord_loss(self, y_true, y_pred):
        b_xy_pred = y_pred[..., :2]
        b_wh_pred = y_pred[..., 2:4]

        b_xy_true = y_true[..., 0:2]
        b_wh_true = y_true[..., 2:4]

        indicator_coord = K.expand_dims(y_true[..., 4], axis=-1)

        loss_xy = K.sum(K.square(b_xy_true - b_xy_pred) * indicator_coord) * self.lambda_coord
        loss_wh = K.sum(K.square(K.sqrt(b_wh_true) - K.sqrt(b_wh_pred)) * indicator_coord) * self.lambda_coord

        return loss_wh + loss_xy

    def obj_loss(self, y_true, y_pred):
        # TODO: should make a review in this part
        obj_conf_true = y_true[..., 4]
        obj_conf_pred = y_pred[..., 4]

        num_true_labels = self.grid_size[0] * self.grid_size[1] * self.nb_anchors
        y_true_coords = K.reshape(y_true[..., :4], shape=(self.batch_size, 1, 1, 1, num_true_labels, 4))
        iou_scores_buff = calculate_ious(y_true_coords, K.expand_dims(y_pred, axis=4))
        best_ious = K.max(iou_scores_buff, axis=4)

        indicator_noobj = K.cast(best_ious < self.iou_filter, np.float32) * (1 - y_true[..., 4]) * self.lambda_noobj
        indicator_obj = y_true[..., 4] * self.lambda_obj
        indicator_obj_noobj = indicator_obj + indicator_noobj

        loss_obj = K.sum(K.square(obj_conf_true - obj_conf_pred) * indicator_obj_noobj)
        return loss_obj

    def class_loss(self, y_true, y_pred):
        # TODO: should we use focal loss?
        p_c_pred = K.softmax(y_pred[..., 5:])
        p_c_true = K.one_hot(K.argmax(y_true[..., 5:], axis=-1), 1)
        loss_class_arg = K.sum(K.square(p_c_true - p_c_pred), axis=-1)

        indicator_class = y_true[..., 4] * self.lambda_class

        loss_class = K.sum(loss_class_arg * indicator_class)

        return loss_class

    def l_coord(self, y_true, y_pred_raw):
        return self.coord_loss(y_true, self._transform_netout(y_pred_raw))

    def l_obj(self, y_true, y_pred_raw):
        return self.obj_loss(y_true, self._transform_netout(y_pred_raw))

    def l_class(self, y_true, y_pred_raw):
        return self.class_loss(y_true, self._transform_netout(y_pred_raw))

    def __call__(self, y_true, y_pred_raw):
        y_pred = self._transform_netout(y_pred_raw)

        total_coord_loss = self.coord_loss(y_true, y_pred)
        total_obj_loss = self.obj_loss(y_true, y_pred)
        total_class_loss = self.class_loss(y_true, y_pred)

        loss = total_coord_loss + total_obj_loss + total_class_loss

        return loss
