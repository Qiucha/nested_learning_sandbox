import torch
from torch.optim import Optimizer

def newton_schulz_iteration(G, steps=5):
    """
    Computes the orthogonalization of 2D tensor G using Newton-Schulz.
    """
    assert len(G.shape) == 2, "Newton-Schulz requires 2D tensors."
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Cast to float32 for stable computation
    X = G.float()
    X = X / (X.norm() + 1e-7)
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.to(G.dtype)

def muon_update(grad, momentum, beta=0.95, ns_steps=5):
    """Applies momentum and orthogonalizes 2D tensors."""
    momentum.mul_(beta).add_(grad, alpha=1 - beta)
    
    if len(grad.shape) >= 2:
        # 2D Parameters: Apply Newton-Schulz
        orig_shape = momentum.shape
        m_2d = momentum.view(orig_shape[0], -1)
        update_2d = newton_schulz_iteration(m_2d, steps=ns_steps)
        return update_2d.view(orig_shape)
    else:
        # 1D Parameters (Biases): Standard normalized momentum fallback
        return momentum / (momentum.norm() + 1e-8)

class Muon(Optimizer):
    """
    Single-device implementation of the Muon optimizer.
    """
    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.0, ns_steps=5):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                
                # Initialization
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                
                # Get the update direction
                update = muon_update(
                    p.grad, 
                    state["momentum_buffer"], 
                    beta=group["momentum"], 
                    ns_steps=group["ns_steps"]
                )
                
                # Apply weight decay if specified
                if group["weight_decay"] > 0.0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                
                # Apply update
                p.add_(update, alpha=-group["lr"])
                
        return loss