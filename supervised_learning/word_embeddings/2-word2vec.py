#!/usr/bin/env python3
"""
2-word2vec module
"""

from gensim.models import Word2Vec


def word2vec_model(
    sentences,
    vector_size=100,
    min_count=5,
    window=5,
    negative=5,
    cbow=True,
    epochs=5,
    seed=0,
    workers=1
):
    """
    funcion documentada
    """
    model = Word2Vec(
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
