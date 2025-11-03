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

        in_h, in_w = 416, 416

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

            anchors_i = self.anchors[i].astype(np.float32)
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        funcion documentada
        """
        fb, fc, fs = [], [], []
        for b, conf, cls in zip(boxes, box_confidences, box_class_probs):
            scores = conf * cls
            classes = np.argmax(scores, axis=-1)
            best_scores = np.max(scores, axis=-1)
            mask = best_scores >= self.class_t
            if np.any(mask):
                fb.append(b[mask])
                fc.append(classes[mask])
                fs.append(best_scores[mask])

        if len(fb) == 0:
            return (np.empty((0, 4)),
                    np.empty((0,), dtype=int),
                    np.empty((0,)))

        filtered_boxes = np.concatenate(fb, axis=0)
        box_classes = np.concatenate(fc, axis=0)
        box_scores = np.concatenate(fs, axis=0)
        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        funcion documentada
        """
        if filtered_boxes.size == 0:
            return (np.empty((0, 4)),
                    np.empty((0,), dtype=int),
                    np.empty((0,)))

        def iou(box, boxes):
            """
            funcion documentada
            """
            x1 = np.maximum(box[0], boxes[:, 0])
            y1 = np.maximum(box[1], boxes[:, 1])
            x2 = np.minimum(box[2], boxes[:, 2])
            y2 = np.minimum(box[3], boxes[:, 3])

            inter_w = np.maximum(0.0, x2 - x1)
            inter_h = np.maximum(0.0, y2 - y1)
            inter = inter_w * inter_h

            area_box = np.maximum(0.0, (box[2] - box[0])) * np.maximum(0.0, (box[3] - box[1]))
            area_boxes = (
                np.maximum(0.0, (boxes[:, 2] - boxes[:, 0])) *
                np.maximum(0.0, (boxes[:, 3] - boxes[:, 1]))
            )
            union = area_box + area_boxes - inter + 1e-16
            return inter / union

        kept_boxes = []
        kept_classes = []
        kept_scores = []

        for c in np.unique(box_classes):
            idxs = np.where(box_classes == c)[0]
            b = filtered_boxes[idxs]
            s = box_scores[idxs]

            order = np.argsort(-s)
            b = b[order]
            s = s[order]

            keep = []
            while order.size > 0:
                keep.append(order[0])
                if order.size == 1:
                    break
                ious = iou(b[0], b[1:])
                remain = np.where(ious <= self.nms_t)[0] + 1
                b = b[remain]
                s = s[remain]
                order = order[remain]

            kept_boxes.append(filtered_boxes[idxs][keep])
            kept_scores.append(box_scores[idxs][keep])
            kept_classes.append(np.full(len(keep), c, dtype=int))

        if len(kept_boxes) == 0:
            return (np.empty((0, 4)),
                    np.empty((0,), dtype=int),
                    np.empty((0,)))

        box_predictions = np.concatenate(kept_boxes, axis=0)
        predicted_box_classes = np.concatenate(kept_classes, axis=0)
        predicted_box_scores = np.concatenate(kept_scores, axis=0)

        order = np.lexsort((-predicted_box_scores, predicted_box_classes))
        box_predictions = box_predictions[order]
        predicted_box_classes = predicted_box_classes[order]
        predicted_box_scores = predicted_box_scores[order]

        return box_predictions, predicted_box_classes, predicted_box_scores
