import torch.optim as optim
from .m3 import M3
from .muon import Muon

def get_optimizer(model, opt_name, lr=1e-3, f=20, stabilize=True):
    """Instantiates the requested optimizer."""
    if opt_name == 'SGD':
        return optim.SGD(model.parameters(), lr=lr)
    elif opt_name == 'MSGD':
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                # Fast Memory
                {'params': model.fast_memory.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': False, 'use_variance': False, 'stabilize': stabilize},

                # Medium Memory
                {'params': model.medium_memory.parameters(), 'alpha': 0.3, 'f': f//2, 'use_muon': False, 'use_variance': False, 'stabilize': stabilize},
                
                # Slow Memory
                {'params': model.slow_memory.parameters(), 'alpha': 0.5, 'f': f, 'use_muon': False, 'use_variance': False, 'stabilize': stabilize}, 
                
                # Isolated Head: Standard Adam updates (No Muon)
                {'params': model.head.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': False, 'use_variance': False, 'stabilize': stabilize}
            ]
            return M3(param_groups, lr=lr)
        else: # FALLBACK FOR BASELINE MODEL
            return M3(model.parameters(), lr=lr, alpha=1.0, use_muon=False, use_variance=False, stabilize = stabilize)
    elif opt_name == 'Adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'MAdam':
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                # Fast Memory
                {'params': model.fast_memory.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': False, 'use_variance': True, 'stabilize': stabilize},

                # Medium Memory
                {'params': model.medium_memory.parameters(), 'alpha': 0.3, 'f': f//2, 'use_muon': False, 'use_variance': True, 'stabilize': stabilize},
                
                # Slow Memory
                {'params': model.slow_memory.parameters(), 'alpha': 0.5, 'f': f, 'use_muon': False, 'use_variance': True, 'stabilize': stabilize}, 
                
                # Isolated Head: Standard Adam updates (No Muon)
                {'params': model.head.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': False, 'use_variance': True, 'stabilize': stabilize}
            ]
            return M3(param_groups, lr=lr)
        else: # FALLBACK FOR BASELINE MODEL
            return M3(model.parameters(), lr=lr, alpha=1.0, use_muon=False, use_variance=True, stabilize = stabilize)
    elif opt_name == 'Muon':
        return Muon(model.parameters(), lr=lr)
    elif opt_name == 'M3S': # Stands for stablized M3 optimizer
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                # Fast Memory
                {'params': model.fast_memory.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': True, 'use_variance': True, 'stabilize': True},

                # Medium Memory
                {'params': model.medium_memory.parameters(), 'alpha': 0.3, 'f': f//2, 'use_muon': True, 'use_variance': True, 'stabilize': True},
                
                # Slow Memory
                {'params': model.slow_memory.parameters(), 'alpha': 0.5, 'f': f, 'use_muon': True, 'use_variance': True, 'stabilize': True}, 
                
                # Isolated Head: Standard Adam updates (No Muon)
                {'params': model.head.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': False, 'use_variance': True, 'stabilize': True}
            ]
            return M3(param_groups, lr=lr)
        else: # FALLBACK FOR BASELINE MODE
            return M3(model.parameters(), lr=lr, alpha=1.0, use_muon=True, use_variance=True, stabilize=True)
    elif opt_name == 'M3': # Unstabilized Version, the same as the original paper
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                # Fast Memory
                {'params': model.fast_memory.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': True, 'use_variance': True, 'stabilize': False},

                # Medium Memory
                {'params': model.medium_memory.parameters(), 'alpha': 0.3, 'f': f//2, 'use_muon': True, 'use_variance': True, 'stabilize': False},
                
                # Slow Memory
                {'params': model.slow_memory.parameters(), 'alpha': 0.5, 'f': f, 'use_muon': True, 'use_variance': True, 'stabilize': False}, 
                
                # Isolated Head: Standard Adam updates (No Muon)
                {'params': model.head.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': False, 'use_variance': True, 'stabilize': False}
            ]
            return M3(param_groups, lr=lr)
        else: # FALLBACK FOR BASELINE MODEL
            return M3(model.parameters(), lr=lr, alpha=1.0, use_muon=True, use_variance=True, stabilize=False)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")