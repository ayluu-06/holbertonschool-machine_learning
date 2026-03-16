#!/usr/bin/env python3
"""
modulo documentado
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    funcion documentada
    """
    rendered_outputs = []
    total_rewards = 0

    state, _ = env.reset()
    rendered_outputs.append(env.render())

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)

        total_rewards += reward
        rendered_outputs.append(env.render())

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
