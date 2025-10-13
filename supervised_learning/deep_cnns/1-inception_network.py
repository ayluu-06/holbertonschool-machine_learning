#!/usr/bin/env python3
"""
modelo documentado
"""

from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    funcion documentada
    """
    X = K.Input(shape=(224, 224, 3))

    x = K.layers.Conv2D(
        64, (7, 7), strides=2, padding='same', activation='relu')(X)
    x = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    x = K.layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x = K.layers.Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    x = inception_block(x, (64, 96, 128, 16, 32, 32))
    x = inception_block(x, (128, 128, 192, 32, 96, 64))
    x = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    x = inception_block(x, (192, 96, 208, 16, 48, 64))
    x = inception_block(x, (160, 112, 224, 24, 64, 64))
    x = inception_block(x, (128, 128, 256, 24, 64, 64))
    x = inception_block(x, (112, 144, 288, 32, 64, 64))
    x = inception_block(x, (256, 160, 320, 32, 128, 128))
    x = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    x = inception_block(x, (256, 160, 320, 32, 128, 128))
    x = inception_block(x, (384, 192, 384, 48, 128, 128))

    x = K.layers.AveragePooling2D(pool_size=(7, 7), strides=1)(x)
    x = K.layers.Dropout(0.4)(x)
    x = K.layers.Dense(1000, activation='softmax')(x)

    return K.models.Model(inputs=X, outputs=x)
