#!/usr/bin/env python3

"""
modulo documentado
"""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """clase documentada"""

    def __init__(self):
        """funcion documentada"""
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """funcion documentada"""
        base_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        base_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased')

        def pt_iterator():
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def en_iterator():
            for _, en in data:
                yield en.numpy().decode('utf-8')

        tokenizer_pt = base_pt.train_new_from_iterator(
            pt_iterator(),
            vocab_size=2 ** 13
        )
        tokenizer_en = base_en.train_new_from_iterator(
            en_iterator(),
            vocab_size=2 ** 13
        )
        return tokenizer_pt, tokenizer_en
