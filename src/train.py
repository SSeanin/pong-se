import torch

from env import PongEnvironment
from agent import DuelingDQNAgent
from buffer import ReplayBuffer
from trainer import Trainer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    capacity = 35000
    batch_size = 64
    target_update_freq = 30000
    max_episodes = 1000
    update_freq = 4

    # Create the environment.
    # Adjust the import as needed.
    env = PongEnvironment(color_reduction_mode='B',
                          frame_stacking=3, frame_skip=1, downsample_size=(84, 84))

    # Get the agent names from the environment.
    agent_names = env.get_agents()

    # Retrieve the observation and action space for initializing networks.
    # Expected to be (channels, height, width)
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n  # type: ignore

    agents = {}
    replay_buffers = {}
    for agent in agent_names:
        agents[agent] = DuelingDQNAgent(name=agent,
                                        input_shape=(3, 84, 84),
                                        num_actions=num_actions,
                                        device=device,
                                        lr=1e-4,
                                        gamma=0.99,
                                        epsilon_start=1.0,
                                        epsilon_final=0.1,
                                        epsilon_decay=200000)
        replay_buffers[agent] = ReplayBuffer(
            capacity=capacity, state_shape=state_shape, device=device)

    # Create the trainer and run training.
    trainer = Trainer(env=env,
                      agents=agents,
                      replay_buffers=replay_buffers,
                      batch_size=batch_size,
                      target_update_freq=target_update_freq,
                      max_episodes=max_episodes,
                      max_steps_per_episode=10000,
                      update_freq=update_freq,
                      device=device)

    episode_rewards = trainer.run()

    # (Optional) Save trained models or plot episode rewards.
    for agent_name, agent in agents.items():
        torch.save(agent.online_net.state_dict(),
                   f"{agent_name}_dueling_dqn.pth")
    print("Training complete!")
