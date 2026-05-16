import torch.nn as nn

class BaselineMLP(nn.Module):
    """A standard, single-head Multi-Layer Perceptron for Class-IL."""
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes) 
        )

    def forward(self, x):
        return self.net(x)