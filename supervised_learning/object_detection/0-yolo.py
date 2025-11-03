#!/usr/bin/env python3
"""
modulo documentado
"""
from tensorflow import keras as K


class Yolo:
    """
    clase documentada
    """

    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        """
        funcion documentada
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, "r") as f:
            self.class_names = [
                line.strip() for line in f if line.strip()]

        self.class_t = float(class_t)
        self.nms_t = float(nms_t)
        self.anchors = anchors
