import torch.nn as nn
from torch.nn import functional as F


class DuelingDQNNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        """
        Dueling DQN network with convolutional feature extractor and two streams
        for value and advantage.

        Args:
            input_shape (tuple): Shape of the input (channels, height, width).
            num_actions (int): Number of possible actions.
        """
        super(DuelingDQNNetwork, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        # Convolutional feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculate the size of the output from the conv layers
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        convw = conv2d_size_out(
            conv2d_size_out(
                conv2d_size_out(input_shape[1], kernel_size=8, stride=4),
                kernel_size=4, stride=2
            ),
            kernel_size=3, stride=1
        )
        convh = conv2d_size_out(
            conv2d_size_out(
                conv2d_size_out(input_shape[2], kernel_size=8, stride=4),
                kernel_size=4, stride=2
            ),
            kernel_size=3, stride=1
        )
        linear_input_size = convw * convh * 64

        # Value stream: estimates the value of being in a given state
        self.value_stream = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Advantage stream: estimates the advantage of each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width)

        Returns:
            torch.Tensor: Q-values for each action, shape (batch, num_actions)
        """
        x = x.permute(0, 3, 1, 2)
        x = x / 255.0

        batch_size = x.size(0)
        features = self.feature_extractor(x)
        features = features.reshape(batch_size, -1)

        # Compute value and advantage streams
        value = self.value_stream(features)            # shape: (batch, 1)
        # shape: (batch, num_actions)
        advantage = self.advantage_stream(features)

        # Combine streams to get Q-values: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
