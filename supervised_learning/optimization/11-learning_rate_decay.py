#!/usr/bin/env python3
"""
Modulo documentado
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    funcion documentada
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
