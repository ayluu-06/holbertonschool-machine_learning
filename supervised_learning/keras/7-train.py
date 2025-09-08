#!/usr/bin/env python3
"""
modelo documentado
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    funcion documentada
    """
    callbacks = []
    if early_stopping and validation_data is not None:
        callbacks.append(K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        ))
    if learning_rate_decay and validation_data is not None:
        def schedule(epoch):
            lr = alpha / (1 + decay_rate * epoch)
            print(
                "\nEpoch {}: LearningRateScheduler setting learning rate to "
                "{}.".format(epoch + 1, lr)
            )
            return lr
        callbacks.append(
            K.callbacks.LearningRateScheduler(schedule, verbose=1))
    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks if callbacks else None
    )
    return history
