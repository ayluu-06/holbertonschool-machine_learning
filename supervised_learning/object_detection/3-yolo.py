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
            return np.empty((0, 4)), np.empty((0,), dtype=int), np.empty((0,))
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

            area_box = (box[2] - box[0]) * (box[3] - box[1])
            area_boxes = (boxes[:, 2] - boxes[:, 0]) * (
                boxes[:, 3] - boxes[:, 1])

            union = area_box + area_boxes - inter + 1e-16
            return inter / union

        keep_boxes = []
        keep_classes = []
        keep_scores = []

        unique_classes = np.unique(box_classes)
        for c in unique_classes:
            idxs = np.where(box_classes == c)[0]
            b = filtered_boxes[idxs]
            s = box_scores[idxs]

            order = np.argsort(-s)
            b = b[order]
            s = s[order]
            idxs_ord = idxs[order]

            selected = []
            while len(b) > 0:
                selected.append(0)
                if len(b) == 1:
                    break
                ious = iou(b[0], b[1:])
                keep = np.where(ious <= self.nms_t)[0] + 1
                b = b[keep]
                s = s[keep]
                idxs_ord = idxs_ord[keep]

            keep_idx = (np.array(idxs)[np.argsort(-s)]
                        if len(selected) == 0 else np.array(idxs)[0:0])

            sel_mask = np.zeros(len(order), dtype=bool)
            sel_mask[np.array(selected)] = True

            b = filtered_boxes[idxs][order][sel_mask]
            s = box_scores[idxs][order][sel_mask]

            keep_boxes.append(b)
            keep_scores.append(s)
            keep_classes.append(np.full(b.shape[0], c, dtype=int))

        if len(keep_boxes) == 0:
            return (np.empty((0, 4)),
                    np.empty((0,), dtype=int),
                    np.empty((0,)))

        box_predictions = np.concatenate(keep_boxes, axis=0)
        predicted_box_classes = np.concatenate(keep_classes, axis=0)
        predicted_box_scores = np.concatenate(keep_scores, axis=0)

        order = np.lexsort((-predicted_box_scores, predicted_box_classes))
        box_predictions = box_predictions[order]
        predicted_box_classes = predicted_box_classes[order]
        predicted_box_scores = predicted_box_scores[order]

        return box_predictions, predicted_box_classes, predicted_box_scores
