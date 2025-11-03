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
        funcion docuemntada
        """
        image_h, image_w = image_size
        in_h = int(self.model.input_shape[1])
        in_w = int(self.model.input_shape[2])

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            gh, gw, nb, _ = output.shape

            t_xy = output[..., 0:2]
            t_wh = output[..., 2:4]
            t_obj = output[..., 4:5]
            t_cls = output[..., 5:]

            cx = np.tile(np.arange(gw).reshape(1, gw, 1), (gh, 1, nb))
            cy = np.tile(np.arange(gh).reshape(gh, 1, 1), (1, gw, nb))

            anchors_i = self.anchors[i]
            pw = anchors_i[:, 0].reshape((1, 1, nb))
            ph = anchors_i[:, 1].reshape((1, 1, nb))

            bx = (sigmoid(t_xy[..., 0]) + cx) / gw
            by = (sigmoid(t_xy[..., 1]) + cy) / gh
            bw = (pw * np.exp(t_wh[..., 0])) / in_w
            bh = (ph * np.exp(t_wh[..., 1])) / in_h

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            boxes.append(np.stack((x1, y1, x2, y2), axis=-1))
            box_confidences.append(sigmoid(t_obj))
            box_class_probs.append(sigmoid(t_cls))

        return boxes, box_confidences, box_class_probs
