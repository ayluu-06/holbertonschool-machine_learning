#!/usr/bin/env python3
"""
modulo documentado
"""

import os
import tensorflow as tf
import tensorflow_hub as hub


_EMBEDDER = hub.load(
    "https://tfhub.dev/google/universal-sentence-encoder-large/5"
)


def _cosine_similarity(a, b):
    """
    funcion documentada
    """
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)

    a_norm = tf.norm(a)
    b_norm = tf.norm(b)

    denom = tf.maximum(a_norm * b_norm, 1e-12)
    return tf.tensordot(a, b, axes=1) / denom


def semantic_search(corpus_path, sentence):
    """
    funcion documentada
    """
    files = [
        f for f in os.listdir(corpus_path)
        if f.endswith(".md")
    ]
    files.sort()

    documents = []
    for fname in files:
        fpath = os.path.join(corpus_path, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            documents.append(f.read())

    if not documents:
        return ""

    embeddings = _EMBEDDER([sentence] + documents)
    query_emb = embeddings[0]
    doc_embs = embeddings[1:]

    best_idx = 0
    best_score = _cosine_similarity(query_emb, doc_embs[0])

    for i in range(1, len(documents)):
        score = _cosine_similarity(query_emb, doc_embs[i])
        if score > best_score:
            best_score = score
            best_idx = i

    return documents[best_idx]
