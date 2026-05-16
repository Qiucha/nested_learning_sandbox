import torch.nn as nn

class CMS_MLP(nn.Module):
    """
    Continuum Memory System MLP.
    Splits the network into working memory (fast), continuum memory (slow),
    and an isolated classification head (fast) to prevent orthogonal wipeout.
    """
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        
        # Fast / Standard updating layers (Shallow features)
        self.working_memory = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Slow / CMS updating layers (Deep persistent features)
        self.continuum_memory = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Fast / Standard updating classification head
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        working_features = self.working_memory(x)
        continuum_features = self.continuum_memory(working_features)
        output = self.head(continuum_features)
        return output