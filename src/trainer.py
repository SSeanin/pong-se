import torch


class Trainer:
    def __init__(self, env, agents, replay_buffers, batch_size=32, target_update_freq=1000,
                 max_episodes=1000, max_steps_per_episode=100000, update_freq=10, device=torch.device("cpu")):
        """
        Trainer to run self-play training for multi-agent Pong with Dueling DQN.

        Args:
            env: An instance of your PongEnvironment.
            agents (dict): A dictionary mapping agent IDs to DuelingDQNAgent instances.
            replay_buffers (dict): A dictionary mapping agent IDs to ReplayBuffer instances.
            batch_size (int): Batch size for updates.
            target_update_freq (int): Frequency (in steps) at which to update target networks.
            max_episodes (int): Maximum number of episodes to run.
            max_steps_per_episode (int): Maximum steps per episode.
            update_freq (int): Frequency (in steps) at which to update the online networks.
            device (torch.device): Device for computation.
        """
        self.env = env
        self.agents = agents
        self.replay_buffers = replay_buffers
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.update_freq = update_freq
        self.device = device

        self.global_step = 0
        self.agent_losses = {}

    def run(self):
        """
        Runs the training loop.
        """
        # For logging purposes, store per-episode rewards for each agent.
        episode_rewards = {agent: [] for agent in self.agents.keys()}

        for episode in range(1, self.max_episodes + 1):
            log = ''
            # Reset the environment. obs is a dict mapping agent -> observation.
            obs, _ = self.env.reset()
            # Initialize per-episode reward trackers.
            episode_reward = {agent: 0.0 for agent in self.agents.keys()}
            # Keep track of done status for each agent.
            done_flags = {agent: False for agent in self.agents.keys()}

            for step in range(self.max_steps_per_episode):
                actions = {}
                # Select an action for each agent that is not yet done.
                for agent, agent_obj in self.agents.items():
                    if not done_flags[agent]:
                        state = obs[agent]
                        action = agent_obj.select_action(state)
                        actions[agent] = action

                # Step the environment.
                next_obs, rewards, terminated, truncated, infos = self.env.step(
                    actions)

                # Store transitions and update episode rewards.
                for agent, agent_obj in self.agents.items():
                    if agent in actions:
                        self.replay_buffers[agent].add(
                            obs[agent],
                            actions[agent],
                            rewards[agent],
                            next_obs[agent],
                            terminated[agent] or truncated[agent]
                        )
                        episode_reward[agent] += rewards[agent]

                obs = next_obs
                self.global_step += 1

                # Update networks every `update_freq` steps if there is enough data.
                if self.global_step % self.update_freq == 0:
                    for agent, agent_obj in self.agents.items():
                        if len(self.replay_buffers[agent]) >= self.batch_size:
                            batch = self.replay_buffers[agent].sample(
                                self.batch_size)
                            loss = agent_obj.update(batch)
                            self.agent_losses[agent] = loss

                # Update target networks periodically.
                if self.global_step % self.target_update_freq == 0:
                    for agent, agent_obj in self.agents.items():
                        agent_obj.update_target()

                # Update done flags.
                for agent in self.agents.keys():
                    done_flags[agent] = terminated[agent] or truncated[agent]

                # If all agents are done, end the episode.
                if all(done_flags.values()):
                    log += f'\nEpisode done after {step} steps\n'
                    log += f'Global steps: {self.global_step}\n'
                    log += f'Agent losses: {self.agent_losses}\n'
                    for agent in self.agents.values():
                        log += f'Current epsilon {agent.name}: {agent.current_epsilon}\n'

                    break

            # Log episode rewards.
            for agent in self.agents.keys():
                episode_rewards[agent].append(episode_reward[agent])

            log = log + f"Episode {episode} | Rewards: {episode_reward}\n"
            with open('run.log', 'a') as log_file:
                log_file.write(log)
            print(log)

            if episode % 50 == 0:
                for agent in self.agents.values():
                    torch.save(agent.online_net.state_dict(),
                               f'checkpoints/{agent.name}-{episode}.pth')

        return episode_rewards
