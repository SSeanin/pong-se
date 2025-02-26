import torch


class ReplayBuffer:
    def __init__(self, capacity, state_shape, device=torch.device("cpu")):
        """
        Replay Buffer implemented as a circular buffer using PyTorch tensors.

        Args:
            capacity (int): Maximum number of transitions to store.
            state_shape (tuple): Shape of the state (e.g., (channels, height, width)).
            device (torch.device): Device where tensors will be stored.
        """
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0

        # Pre-allocate memory for each component of the transition.
        self.states = torch.empty(
            (capacity, *state_shape), dtype=torch.float32)
        self.next_states = torch.empty(
            (capacity, *state_shape), dtype=torch.float32)
        # Assuming discrete actions.
        self.actions = torch.empty(
            (capacity,), dtype=torch.int64)
        self.rewards = torch.empty(
            (capacity,), dtype=torch.float32)
        self.dones = torch.empty((capacity,), dtype=torch.bool)

    def add(self, state, action, reward, next_state, done):
        """
        Adds a new transition to the replay buffer. If the buffer is full, it overwrites the oldest transition.

        Args:
            state (torch.Tensor or array-like): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (torch.Tensor or array-like): The next state.
            done (bool): Whether the episode has ended.
        """
        # If state or next_state are not tensors, convert them.
        if not torch.is_tensor(state):
            state = torch.tensor(
                state, dtype=torch.float32)
        if not torch.is_tensor(next_state):
            next_state = torch.tensor(
                next_state, dtype=torch.float32)

        self.states[self.pos].copy_(state)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos].copy_(next_state)
        self.dones[self.pos] = done

        # Move pointer in a circular manner.
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Samples a random batch of transitions from the replay buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            A tuple of (states, actions, rewards, next_states, dones) where each is a PyTorch tensor.
        """
        # Randomly choose indices
        indices = torch.randint(0, self.size, (batch_size,))

        batch_states = self.states[indices].to(device=self.device)
        batch_actions = self.actions[indices].to(device=self.device)
        batch_rewards = self.rewards[indices].to(device=self.device)
        batch_next_states = self.next_states[indices].to(device=self.device)
        batch_dones = self.dones[indices].to(device=self.device)

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones

    def __len__(self):
        """Returns the current size of the buffer."""
        return self.size
