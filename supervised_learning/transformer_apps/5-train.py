#!/usr/bin/env python3
"""modulo documentado"""

import tensorflow as tf

Dataset = __import__("3-dataset").Dataset
create_masks = __import__("4-create_masks").create_masks
Transformer = __import__("5-transformer").Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """clase documentada"""

    def __init__(self, dm, warmup_steps=4000):
        """funcion documentada"""
        super().__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        """funcion documentada"""
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * tf.pow(self.warmup_steps, -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def _loss_function(y_true, y_pred):
    """funcion documentada"""
    scce = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    loss = scce(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def _accuracy_function(y_true, y_pred):
    """funcion documentada"""
    y_hat = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
    matches = tf.cast(tf.equal(y_true, y_hat), tf.float32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    matches *= mask
    return tf.reduce_sum(matches) / tf.reduce_sum(mask)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    funcion documentada
    """
    data = Dataset(batch_size, max_len)

    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(
        N=N,
        dm=dm,
        h=h,
        hidden=hidden,
        input_vocab=input_vocab,
        target_vocab=target_vocab,
        max_seq_input=max_len,
        max_seq_target=max_len,
    )

    lr = CustomSchedule(dm, warmup_steps=4000)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9,
    )

    for epoch in range(1, epochs + 1):
        epoch_loss = tf.keras.metrics.Mean()
        epoch_acc = tf.keras.metrics.Mean()

        for batch, (inputs, target) in enumerate(data.data_train):
            tar_inp = target[:, :-1]
            tar_real = target[:, 1:]

            enc_mask, combined_mask, dec_mask = create_masks(inputs, tar_inp)

            with tf.GradientTape() as tape:
                preds = transformer(
                    inputs,
                    tar_inp,
                    True,
                    enc_mask,
                    combined_mask,
                    dec_mask,
                )
                loss = _loss_function(tar_real, preds)

            grads = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, transformer.trainable_variables))

            acc = _accuracy_function(tar_real, preds)
            epoch_loss.update_state(loss)
            epoch_acc.update_state(acc)

            if batch % 50 == 0:
                print(
                    "Epoch {}, Batch {}: Loss {}, Accuracy {}".format(
                        epoch,
                        batch,
                        float(epoch_loss.result()),
                        float(epoch_acc.result()),
                    )
                )

        print(
            "Epoch {}: Loss {}, Accuracy {}".format(
                epoch,
                float(epoch_loss.result()),
                float(epoch_acc.result()),
            )
        )

    return transformer
