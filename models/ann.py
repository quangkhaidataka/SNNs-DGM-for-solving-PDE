import torch
import torch.nn as nn
import numpy as np


# Neural Network Definition
class DeepGalerkinNet(nn.Module):
    def __init__(self, d=2, num_hidden_layers=4, num_hidden_units=50):
        super(DeepGalerkinNet, self).__init__()
        self.d = d  # Number of spatial dimensions
        self.layers = nn.ModuleList()
        # Input layer: (t, x_1, ..., x_d) -> num_hidden_units
        self.layers.append(nn.Linear(1 + d, num_hidden_units))
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(num_hidden_units, num_hidden_units))
        # Output layer: num_hidden_units -> 1 (u(t,x))
        self.layers.append(nn.Linear(num_hidden_units, 1))
        self.activation = nn.Tanh()

    def forward(self, inputs):
        # t: (batch_size, 1), x: (batch_size, d)
        # inputs = torch.cat([t, x], dim=1)  # Shape: (batch_size, 1 + d)
        for i, layer in enumerate(self.layers):
            inputs = layer(inputs)
            if i < len(self.layers) - 1:  # Apply activation except for last layer
                inputs = self.activation(inputs)
        return inputs
