#!/usr/bin/env python3
"""modulo documentado"""

import tensorflow as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """clase documentada"""

    def __init__(self):
        """funcion documentada"""
        raw_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )
        raw_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(raw_train)

        self.data_train = raw_train.map(
            self.tf_encode,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        self.data_valid = raw_valid.map(
            self.tf_encode,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    def tokenize_dataset(self, data):
        """funcion documentada"""
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        pt_texts = []
        en_texts = []

        for pt, en in data:
            pt_texts.append(pt.numpy().decode('utf-8'))
            en_texts.append(en.numpy().decode('utf-8'))

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_texts,
            vocab_size=2 ** 13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_texts,
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """funcion documentada"""
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        pt_ids = self.tokenizer_pt.encode(pt_text, add_special_tokens=False)
        en_ids = self.tokenizer_en.encode(en_text, add_special_tokens=False)

        pt_vocab = self.tokenizer_pt.vocab_size
        en_vocab = self.tokenizer_en.vocab_size

        pt_tokens = [pt_vocab] + pt_ids + [pt_vocab + 1]
        en_tokens = [en_vocab] + en_ids + [en_vocab + 1]

        pt_tokens = tf.convert_to_tensor(pt_tokens, dtype=tf.int64)
        en_tokens = tf.convert_to_tensor(en_tokens, dtype=tf.int64)

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """funcion documentada"""

        def _encode(p, e):
            return self.encode(p, e)

        pt_tokens, en_tokens = tf.py_function(
            func=_encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
