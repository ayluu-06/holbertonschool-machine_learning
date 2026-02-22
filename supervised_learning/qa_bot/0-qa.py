#!/usr/bin/env python3
"""
0-qa.py
"""

import tensorflow as tf
import tensorflow_hub as hub
import transformers


_QA_MODEL_URL = "https://tfhub.dev/see--/bert-uncased-tf2-qa/1"

_tokenizer = transformers.BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)
_model = hub.load(_QA_MODEL_URL)


def question_answer(question, reference):
    """
    funcion documentada
    """
    if question is None or reference is None:
        return None

    question = question.strip()
    reference = reference.strip()
    if not question or not reference:
        return None

    encoded = _tokenizer.encode_plus(
        question,
        reference,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        return_tensors="tf",
    )

    input_word_ids = tf.cast(encoded["input_ids"], tf.int32)
    input_mask = tf.cast(encoded["attention_mask"], tf.int32)
    input_type_ids = tf.cast(encoded["token_type_ids"], tf.int32)

    outputs = _model(
        {
            "input_word_ids": input_word_ids,
            "input_mask": input_mask,
            "input_type_ids": input_type_ids,
        }
    )

    start_logits = outputs["start_logits"][0]
    end_logits = outputs["end_logits"][0]

    start_idx = int(tf.argmax(start_logits, axis=0).numpy())
    end_idx = int(tf.argmax(end_logits, axis=0).numpy())

    if end_idx < start_idx:
        return None

    input_ids = encoded["input_ids"][0].numpy().tolist()
    answer_ids = input_ids[start_idx: end_idx + 1]
    answer_tokens = _tokenizer.convert_ids_to_tokens(answer_ids)
    answer = _tokenizer.convert_tokens_to_string(answer_tokens).strip()

    if not answer:
        return None

    bad = {"[CLS]", "[SEP]"}
    if answer in bad:
        return None

    return answer
