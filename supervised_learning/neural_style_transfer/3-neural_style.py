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
        self.alpha = alpha
        self.beta = beta
        self.model = None

        self.style_features = None
        self.gram_style_features = None
        self.content_feature = None

        self.load_model()
        self.generate_features()

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
        scale = 512.0 / max(h, w)
        new_h = max(int(round(h * scale)), 1)
        new_w = max(int(round(w * scale)), 1)

        img = tf.convert_to_tensor(image, dtype=tf.float32)
        img = tf.expand_dims(img, axis=0)
        img = tf.image.resize(
            img, (new_h, new_w), method=tf.image.ResizeMethod.BICUBIC
        )
        img = tf.clip_by_value(img / 255.0, 0.0, 1.0)
        return img

    def load_model(self):
        """
        funcion documentada
        """
        base = tf.keras.applications.vgg19.VGG19(
            include_top=False, weights='imagenet'
        )
        base.trainable = False

        inp = base.input
        x = inp
        name_to_tensor = {}

        for layer in base.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )(x)
            elif isinstance(layer, tf.keras.layers.Conv2D):
                cfg = {
                    "filters": layer.filters,
                    "kernel_size": layer.kernel_size,
                    "strides": layer.strides,
                    "padding": layer.padding,
                    "activation": layer.activation,
                    "use_bias": layer.use_bias,
                    "name": layer.name
                }
                new_conv = tf.keras.layers.Conv2D(**cfg)
                x = new_conv(x)
                new_conv.set_weights(layer.get_weights())
                new_conv.trainable = False
            else:
                x = x
            name_to_tensor[layer.name] = x

        outputs = [name_to_tensor[name] for name in self.style_layers]
        outputs.append(name_to_tensor[self.content_layer])

        self.model = tf.keras.Model(inputs=inp, outputs=outputs)
        self.model.trainable = False

    @staticmethod
    def gram_matrix(input_layer):
        """
        funcion documentada
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError("input_layer must be a tensor of rank 4")
        if input_layer.shape.rank != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape
        h = int(h)
        w = int(w)
        c = int(c)

        feats = tf.reshape(input_layer, (1, h * w, c))
        feats = tf.cast(feats, tf.float32)
        gram = tf.matmul(feats, feats, transpose_a=True)
        gram /= tf.cast(h * w, tf.float32)
        gram = tf.reshape(gram, (1, c, c))
        return gram

    def generate_features(self):
        """
        funcion documentada
        """
        if self.model is None:
            self.load_model()

        style_outputs = self.model(self.style_image)
        content_outputs = self.model(self.content_image)

        style_feats = style_outputs[:-1]
        content_feat = content_outputs[-1]

        self.style_features = style_feats
        self.gram_style_features = [
            self.gram_matrix(feat) for feat in style_feats
        ]
        self.content_feature = content_feat
