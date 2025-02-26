import torch
import gymnasium as gym
import ale_py
import supersuit as ss

from agent import DuelingDQNAgent

gym.register_envs(ale_py)

env = gym.make('PongNoFrameskip-v4', render_mode='human')
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v1(
    env, x_size=84, y_size=84)
env = ss.frame_skip_v0(env, 1)
env = ss.frame_stack_v1(env, 3)

observation, info = env.reset(seed=42)

agent = DuelingDQNAgent('first_0', (3, 84, 84), 6, torch.device(
    'cuda'), epsilon_start=0, epsilon_final=0
)

agent.load_model('checkpoints/run4/first_0-1000.pth')


for _ in range(10000):
    # this is where you would insert your policy
    action = agent.select_action(observation)

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
