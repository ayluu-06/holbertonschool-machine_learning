#!/usr/bin/env python3
"""
modulo documentado
"""

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    funcion documentada
    """
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done and reward == 0:
                reward = -1

            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action]
            )

            state = new_state
            episode_reward += reward

            if done:
                break

        total_rewards.append(episode_reward)

        epsilon = min_epsilon + (epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode
        )

    return Q, total_rewards
