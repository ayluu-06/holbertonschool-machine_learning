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
        img_h, img_w = image_size.astype(np.float32)
        in_h = float(self.model.input_shape[1])
        in_w = float(self.model.input_shape[2])

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        boxes = []
        box_confidences = []
        box_class_probs = []

        n_out = len(outputs)

        for i, out in enumerate(outputs):
            gh, gw, nb, _ = out.shape

            t_xy = out[..., 0:2].astype(np.float32)
            t_wh = out[..., 2:4].astype(np.float32)
            t_obj = out[..., 4:5].astype(np.float32)
            t_cls = out[..., 5:].astype(np.float32)

            gy, gx = np.indices((gh, gw), dtype=np.float32)
            gx = np.expand_dims(gx, axis=-1)
            gy = np.expand_dims(gy, axis=-1)

            anchors_i = self.anchors[n_out - 1 - i].astype(np.float32)
            anchors_i = anchors_i.reshape(1, 1, nb, 2)
            pw = anchors_i[..., 0]
            ph = anchors_i[..., 1]

            bx = (sigmoid(t_xy[..., 0]) + gx) / float(gw)
            by = (sigmoid(t_xy[..., 1]) + gy) / float(gh)

            bw = (pw * np.exp(t_wh[..., 0])) / in_w
            bh = (ph * np.exp(t_wh[..., 1])) / in_h

            x1 = (bx - bw / 2.0) * img_w
            y1 = (by - bh / 2.0) * img_h
            x2 = (bx + bw / 2.0) * img_w
            y2 = (by + bh / 2.0) * img_h

            boxes.append(np.stack((x1, y1, x2, y2), axis=-1))
            box_confidences.append(sigmoid(t_obj))
            box_class_probs.append(sigmoid(t_cls))

        return boxes, box_confidences, box_class_probs
