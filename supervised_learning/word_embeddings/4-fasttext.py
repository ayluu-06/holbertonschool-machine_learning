#!/usr/bin/env python3
"""
4-fasttext module
"""

from gensim.models import FastText


def fasttext_model(
    sentences,
    vector_size=100,
    min_count=5,
    negative=5,
    window=5,
    cbow=True,
    epochs=5,
    seed=0,
    workers=1
):
    """
    funcion documentada
    """
    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=0 if cbow else 1,
        negative=negative,
        seed=seed
    )

    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
