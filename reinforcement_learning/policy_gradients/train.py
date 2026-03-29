#!/usr/bin/env python3
"""
modulo documentado
"""
import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    funcion documentada
    """
    weights = np.random.rand(*env.observation_space.shape, env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        grads = []
        rewards = []
        score = 0
        done = False

        while not done:
            if show_result and episode % 1000 == 0:
                env.render()

            action, grad = policy_gradient(state, weights)
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            grads.append(grad)
            rewards.append(reward)
            score += reward
            state = new_state

        scores.append(score)
        print("Episode: {} Score: {}".format(episode, score))

        for i, grad in enumerate(grads):
            discount = 0
            for j, reward in enumerate(rewards[i:]):
                discount += reward * (gamma ** j)
            weights -= alpha * grad * discount

    return scores
