import torch
from torch.optim import Optimizer
import torch.distributed as dist

def newton_schulz_iteration(G, steps=5):
    """
    Computes the orthogonalization of 2D tensor G using Newton-Schulz.
    (Approximation used in modern Muon implementations).
    """
    assert len(G.shape) == 2, "Newton-Schulz requires 2D tensors."
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Cast to bfloat16 or float32 for stable computation
    X = G.float()
    X = X / (X.norm() + 1e-7)
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.to(G.dtype)

# The following code comes from the implementation in https://github.com/KellerJordan/Muon/blob/master/muon.py
"""
@misc{jordan2024muon,
  author       = {Keller Jordan and Yuchen Jin and Vlado Boza and You Jiacheng and
                  Franz Cesista and Laker Newhouse and Jeremy Bernstein},
  title        = {Muon: An optimizer for hidden layers in neural networks},
  year         = {2024},
  url          = {https://kellerjordan.github.io/posts/muon/}
}
"""

def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = newton_schulz_iteration(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1))**0.5
    return update

class Muon(Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0, momentum=0.95, use_muon=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        defaults = dict(lr=lr, alpha=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
        return loss