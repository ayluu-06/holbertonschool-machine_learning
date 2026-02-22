#!/usr/bin/env python3
"""
modulo documentado
"""

semantic_search = __import__('3-semantic_search').semantic_search
qa_single = __import__('0-qa').question_answer


def question_answer(corpus_path):
    """
    funcion documentada
    """
    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        question = input("Q: ")
        if question.strip().lower() in exit_words:
            print("A: Goodbye")
            return

        reference = semantic_search(corpus_path, question)
        answer = qa_single(question, reference)

        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print("A: {}".format(answer))
