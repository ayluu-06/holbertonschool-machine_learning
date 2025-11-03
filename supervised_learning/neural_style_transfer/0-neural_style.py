#!/usr/bin/env python3
"""
modulo documentado
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    clase documentada
    """

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1',
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        funcion documentada
        """
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = (alpha)
        self.beta = (beta)

    @staticmethod
    def scale_image(image):
        """
        funcion documentada
        """
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w = image.shape[:2]
        if h == 0 or w == 0:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        scale = 512.0 / max(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        new_h = max(new_h, 1)
        new_w = max(new_w, 1)

        img = tf.convert_to_tensor(image, dtype=tf.float32)
        img = tf.expand_dims(img, axis=0)
        img = tf.image.resize(
            img, (new_h, new_w), method=tf.image.ResizeMethod.BICUBIC
        )
        img = img / 255.0
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img
