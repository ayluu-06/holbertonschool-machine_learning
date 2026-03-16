#!/usr/bin/env python3
"""
modulo documentado
"""

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Permute
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


class GymnasiumToKerasRL(gym.Wrapper):
    """clase documentada"""

    def reset(self, **kwargs):
        """funcion documentada"""
        observation, _ = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        """funcion documentada"""
        observation, reward, terminated, truncated, info = self.env.step(
            action
        )
        done = terminated or truncated
        return observation, reward, done, info

    def render(self, mode='human'):
        """funcion documentada"""
        return self.env.render()


def build_model(height, width, channels, actions):
    """funcion documentada"""
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(channels, height, width)))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


def build_agent(model, actions):
    """funcion documentada"""
    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=1000000, window_length=4)

    dqn = DQNAgent(
        model=model,
        nb_actions=actions,
        memory=memory,
        policy=policy,
        nb_steps_warmup=50000,
        target_model_update=10000,
        enable_double_dqn=True,
        batch_size=32,
        gamma=0.99
    )

    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    return dqn


if __name__ == '__main__':
    env = gym.make('ALE/Breakout-v5')
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=True
    )
    env = GymnasiumToKerasRL(env)

    height, width = env.observation_space.shape
    channels = 4
    actions = env.action_space.n

    model = build_model(height, width, channels, actions)
    dqn = build_agent(model, actions)

    dqn.fit(env, nb_steps=500000, visualize=False, verbose=2)

    model.save_weights('policy.h5')
    env.close()
