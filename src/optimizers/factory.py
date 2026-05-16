import torch.optim as optim
from .m3 import M3, SM3
from .muon import Muon

def get_optimizer(model, opt_name, lr=1e-3, f=20):
    """Instantiates the requested optimizer."""
    if opt_name == 'SGD':
        return optim.SGD(model.parameters(), lr=lr)
    elif opt_name == 'Adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'Muon':
        if hasattr(model, 'continuum_memory') and hasattr(model, 'working_memory'):
            param_groups = [
                # Fast Memory: Multi-timescale Newton-Schulz orthogonalization
                {'params': model.working_memory.parameters()},
                
                # Slow Memory: Multi-timescale Newton-Schulz orthogonalization
                {'params': model.continuum_memory.parameters()}, 
                
                # Isolated Head: Standard Adam updates (No Muon)
                {'params': model.head.parameters(), 'use_muon': False}
            ]
            return Muon(param_groups, lr=lr)
        else: # FALLBACK FOR BASELINE MODEL
            return Muon(model.parameters(), lr=lr)
    elif opt_name == 'SM3': # Stands for stablized M3 optimizer
        if hasattr(model, 'continuum_memory') and hasattr(model, 'working_memory'):
            param_groups = [
                # Fast Memory: Multi-timescale Newton-Schulz orthogonalization
                {'params': model.working_memory.parameters(), 'alpha': 0.0, 'f': f//2, 'use_muon': True},
                
                # Slow Memory: Multi-timescale Newton-Schulz orthogonalization
                {'params': model.continuum_memory.parameters(), 'alpha': 0.5, 'f': f, 'use_muon': True}, 
                
                # Isolated Head: Standard Adam updates (No Muon)
                {'params': model.head.parameters(), 'alpha': 0.0, 'f': f//2, 'use_muon': False}
            ]
            return SM3(param_groups, lr=lr)
        else: # FALLBACK FOR BASELINE MODEL
            return SM3(model.parameters(), lr=lr, alpha=1.0)
    elif opt_name == 'M3':
        if hasattr(model, 'continuum_memory') and hasattr(model, 'working_memory'):
            param_groups = [
                # Fast Memory: Multi-timescale Newton-Schulz orthogonalization
                {'params': model.working_memory.parameters(), 'alpha': 0.0, 'f': f//2, 'use_muon': True},
                
                # Slow Memory: Multi-timescale Newton-Schulz orthogonalization
                {'params': model.continuum_memory.parameters(), 'alpha': 0.5, 'f': f, 'use_muon': True}, 
                
                # Isolated Head: Standard Adam updates (No Muon)
                {'params': model.head.parameters(), 'alpha': 0.0, 'f': f//2, 'use_muon': False}
            ]
            return M3(param_groups, lr=lr)
        else: # FALLBACK FOR BASELINE MODEL
            return M3(model.parameters(), lr=lr, alpha=1.0)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")