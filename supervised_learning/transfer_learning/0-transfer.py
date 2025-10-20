#!/usr/bin/env python3
"""
modulo documentado
"""

from tensorflow import keras as K


def preprocess_data(X, Y):
    """
    funcion documentada
    """
    X_p = X.astype("float32")  # a float
    Y_p = K.utils.to_categorical(Y.reshape(-1), 10)
    return X_p, Y_p


def build_frozen_backbone():
    """
    funcion documentada
    """
    inp = K.Input(shape=(32, 32, 3))  # formita y eso
    x = K.layers.Resizing(224, 224)(inp)
    x = K.layers.Lambda(K.applications.densenet.preprocess_input)(x)

    # el densenet preentrenado
    base = K.applications.DenseNet201(include_top=False,
                                      weights="imagenet",
                                      input_shape=(224, 224, 3))
    base.trainable = False  # lo mantenemos congeladito
    x = base(x)
    x = K.layers.GlobalMaxPooling2D()(x)
    return K.Model(inp, x)


def build_head(feature_dim):
    """
    funcion documentada
    """
    he = K.initializers.he_normal()
    l2 = K.regularizers.l2(1e-4)
    fin = K.Input(shape=(feature_dim,))
    h = K.layers.Dense(256, activation="elu",
                       kernel_initializer=he,
                       kernel_regularizer=l2)(fin)
    h = K.layers.Dropout(0.5)(h)
    out = K.layers.Dense(10, activation="softmax",
                         kernel_initializer=he,
                         kernel_regularizer=l2)(h)
    return K.Model(fin, out)


if __name__ == "__main__":
    (X_tr, Y_tr), (X_te, Y_te) = K.datasets.cifar10.load_data()
    X_tr, Y_tr = preprocess_data(X_tr, Y_tr)
    X_te, Y_te = preprocess_data(X_te, Y_te)

    backbone = build_frozen_backbone()  # backbone congeladito y pre calculo
    bs = 32  # solo por lote(?
    tr_gen = K.preprocessing.image.ImageDataGenerator().flow(
        X_tr, Y_tr, batch_size=bs, shuffle=False
    )
    te_gen = K.preprocessing.image.ImageDataGenerator().flow(
        X_te, Y_te, batch_size=bs, shuffle=False
    )
    feats_tr = backbone.predict(tr_gen, verbose=1)  # solo los "importantes"
    feats_te = backbone.predict(te_gen, verbose=1)

    head = build_head(feats_tr.shape[1])
    head.compile(optimizer=K.optimizers.Adam(learning_rate=1e-4),
                 loss="categorical_crossentropy",
                 metrics=["accuracy"])

    # aca un parate si tranca o si no mejora, en ese caso guardar el mejor
    lr_reduce = K.callbacks.ReduceLROnPlateau(monitor="val_accuracy",
                                              factor=0.6,
                                              patience=2,
                                              verbose=1,
                                              mode="max",
                                              min_lr=1e-7)
    early_stop = K.callbacks.EarlyStopping(monitor="val_accuracy",
                                           patience=3,
                                           verbose=1,
                                           mode="max",
                                           restore_best_weights=True)
    checkpoint = K.callbacks.ModelCheckpoint("cifar10.h5",
                                             monitor="val_accuracy",
                                             mode="max",
                                             verbose=1,
                                             save_best_only=True,
                                             save_weights_only=False)

    head.fit(feats_tr, Y_tr,
             validation_data=(feats_te, Y_te),
             epochs=20,
             batch_size=bs,
             shuffle=True,
             verbose=1,
             callbacks=[lr_reduce, early_stop, checkpoint])

    img_in = backbone.input  # todo juntito ahora
    logits = head(backbone.output)
    full_model = K.Model(img_in, logits)

    full_model.compile(optimizer=K.optimizers.Adam(learning_rate=1e-4),
                       loss="categorical_crossentropy",
                       metrics=["accuracy"])
    full_model.save("cifar10.h5")  # ahi guardamos y fin
