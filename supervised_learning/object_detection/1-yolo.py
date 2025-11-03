#!/usr/bin/env python3
"""
modulo documentado
"""
from tensorflow import keras as K
import numpy as np


class Yolo:
    """
    clase documentada
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        funcion documentada
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f if line.strip()]
        self.class_t = float(class_t)
        self.nms_t = float(nms_t)
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        funcion documentada
        """
        image_h, image_w = image_size
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        def sigmoid(x):
            return 1. / (1. + np.exp(-x))

        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            gh, gw, anchor_boxes, _ = output.shape

            t_xy = output[..., 0:2]
            t_wh = output[..., 2:4]
            t_conf = output[..., 4:5]
            t_class = output[..., 5:]

            grid_x = np.arange(gw).reshape(1, gw, 1)
            grid_y = np.arange(gh).reshape(gh, 1, 1)

            bx = (sigmoid(t_xy[..., 0]) + grid_x) / gw
            by = (sigmoid(t_xy[..., 1]) + grid_y) / gh

            anchor_wh = self.anchors[i].reshape(1, 1, anchor_boxes, 2)
            pw = anchor_wh[..., 0]
            ph = anchor_wh[..., 1]

            bw = (pw * np.exp(t_wh[..., 0])) / input_w
            bh = (ph * np.exp(t_wh[..., 1])) / input_h

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            box_confidences.append(sigmoid(t_conf))
            box_class_probs.append(sigmoid(t_class))

        return boxes, box_confidences, box_class_probs
