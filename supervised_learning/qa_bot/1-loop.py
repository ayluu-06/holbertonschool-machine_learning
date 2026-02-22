#!/usr/bin/env python3
"""
modulo documentado
"""


def main():
    """funcion documentada"""
    while True:
        question = input("Q: ")

        if question.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break

        print("A:")


if __name__ == "__main__":
    main()
