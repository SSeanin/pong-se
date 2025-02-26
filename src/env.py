from pettingzoo.atari import pong_v3
import supersuit as ss


class PongEnvironment:
    def __init__(self, color_reduction_mode='B', frame_stacking=3, frame_skip=1, downsample_size=(84, 84)):
        self.downsample_size = downsample_size
        self.frame_stacking = frame_stacking

        self.env = pong_v3.parallel_env(render_mode='rgb_array')
        self.env.reset()

        self.agents = self.env.agents

        self.env = ss.color_reduction_v0(self.env, mode=color_reduction_mode)
        self.env = ss.resize_v1(
            self.env, x_size=downsample_size[0], y_size=downsample_size[1])
        self.env = ss.frame_skip_v0(self.env, frame_skip)
        self.env = ss.frame_stack_v1(self.env, frame_stacking)
        self.observation_space = self.env.observation_space(self.agents[0])
        self.action_space = self.env.action_space(self.agents[0])

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()

    def get_agents(self):
        return self.agents
