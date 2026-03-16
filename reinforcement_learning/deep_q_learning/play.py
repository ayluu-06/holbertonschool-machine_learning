#!/usr/bin/env python3
"""
modulo documentado
"""

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Permute

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy


class GymnasiumToKerasRL(gym.Wrapper):
    """funcion documentada"""

    def reset(self, **kwargs):
        """Return only the observation."""
        observation, _ = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        """
        funcion documentada
        """
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
    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = GreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=actions,
        memory=memory,
        policy=policy,
        nb_steps_warmup=0,
        target_model_update=10000,
        enable_double_dqn=True,
        batch_size=32,
        gamma=0.99
    )

    dqn.compile(optimizer='adam', metrics=['mae'])
    return dqn


if __name__ == '__main__':
    env = gym.make('ALE/Breakout-v5', render_mode='human')
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
    model.load_weights('policy.h5')

    dqn = build_agent(model, actions)
    dqn.test(env, nb_episodes=1, visualize=True)

    env.close()
