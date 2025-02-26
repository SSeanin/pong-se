import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from network import DuelingDQNNetwork


class DuelingDQNAgent:
    def __init__(self, name, input_shape, num_actions, device=torch.device("cpu"),
                 lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_final=0.1, epsilon_decay=100000):
        """
        Dueling DQN Agent.

        Args:
            input_shape (tuple): Shape of the state (channels, height, width).
            num_actions (int): Number of possible actions.
            device (torch.device): Device for computations.
            lr (float): Learning rate.
            gamma (float): Discount factor.
            epsilon_start (float): Initial value of epsilon.
            epsilon_final (float): Minimum value of epsilon.
            epsilon_decay (int): Decay rate for epsilon.
        """
        self.name = name
        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma

        # Initialize the online and target networks.
        self.online_net = DuelingDQNNetwork(
            input_shape, num_actions).to(device)
        self.target_net = DuelingDQNNetwork(
            input_shape, num_actions).to(device)
        # Ensure target network starts with same parameters.
        self.update_target()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # Epsilon-greedy parameters.
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.frame_idx = 0

    def select_action(self, state):
        """
        Selects an action using an epsilon-greedy strategy.

        Args:
            state (np.array or torch.Tensor): Current state of shape (channels, height, width).

        Returns:
            int: The action selected.
        """
        self.frame_idx += 1
        # Decay epsilon exponentially.
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
            np.exp(-1. * self.frame_idx / self.epsilon_decay)

        self.current_epsilon = epsilon

        if np.random.rand() < epsilon:
            # Random action.
            return np.random.randint(0, self.num_actions)
        else:
            # Convert state to tensor if needed.
            if not torch.is_tensor(state):
                state = torch.tensor(
                    state, dtype=torch.float32, device=self.device)
            # Add batch dimension if missing.
            if len(state.shape) == len(self.online_net.input_shape):
                state = state.unsqueeze(0)
            with torch.no_grad():
                q_values = self.online_net(state)
            return int(torch.argmax(q_values, dim=1).item())

    def update(self, batch):
        """
        Updates the online network parameters based on a batch of transitions.

        Args:
            batch (tuple): Tuple of (states, actions, rewards, next_states, dones),
                           where each element is a PyTorch tensor.

        Returns:
            float: The computed loss value.
        """
        states, actions, rewards, next_states, dones = batch

        # Compute Q-values for current states using the online network.
        q_values = self.online_net(states)
        # Gather the Q-values for the taken actions.
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute the target Q-values using the target network.
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_q_value = next_q_values.max(1)[0]
            # Use (1 - done) to zero-out the next state value when the episode ends.
            expected_q_value = rewards + self.gamma * \
                next_q_value * (1 - dones.float())

        # Compute the Mean Squared Error loss.
        loss = nn.MSELoss()(q_value, expected_q_value)

        # Optimize the online network.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """
        Updates the target network by copying parameters from the online network.
        """
        self.target_net.load_state_dict(self.online_net.state_dict())

    def load_model(self, model_path):
        self.online_net.load_state_dict(torch.load(model_path, weights_only=True))
