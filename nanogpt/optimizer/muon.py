"""
Muon optimizer: MomentUm Orthogonalized by Newton-schulz.

Reference: https://github.com/KellerJordan/modded-nanogpt
"""

import torch


# Default Newton-Schulz iteration parameters
DEFAULT_NS_COEFFS = (3.4445, -4.7750, 2.0315)
DEFAULT_NS_STEPS = 5


def make_zeropower_fn(ns_a, ns_b, ns_c, ns_steps=DEFAULT_NS_STEPS):
    """
    Create a compiled Newton-Schulz function with given coefficients.

    Args:
        ns_a, ns_b, ns_c: Newton-Schulz iteration coefficients
        ns_steps: Number of iterations

    Returns:
        Compiled function for orthogonalization
    """

    @torch.compile
    def zeropower_via_newtonschulz5(G, steps=ns_steps, eps=1e-7):
        assert len(G.shape) == 2
        a, b, c = ns_a, ns_b, ns_c
        X = G.bfloat16() / (G.norm() + eps)
        if G.size(0) > G.size(1):
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = A @ X
            X = a * X + b * B + c * A @ B
        if G.size(0) > G.size(1):
            X = X.T
        return X.to(G.dtype)

    return zeropower_via_newtonschulz5


# Default zeropower function (for backward compatibility)
zeropower_via_newtonschulz5 = make_zeropower_fn(*DEFAULT_NS_COEFFS)


class Muon(torch.optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    This optimizer is designed for transformer block parameters (attention, MLP weights).
    For embedding and output layers, use AdamW instead.

    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 3e-4)
        momentum: Momentum factor (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_coeffs: Tuple of (a, b, c) Newton-Schulz coefficients (default: (3.4445, -4.7750, 2.0315))
        ns_steps: Number of Newton-Schulz iterations (default: 5)
        weight_decay: Decoupled weight decay (default: 0.0)
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        momentum=0.95,
        nesterov=True,
        ns_coeffs=DEFAULT_NS_COEFFS,
        ns_steps=DEFAULT_NS_STEPS,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        # Create zeropower function with given coefficients
        self.zeropower_fn = make_zeropower_fn(*ns_coeffs, ns_steps=ns_steps)

    def step(self):
        """Perform a single optimization step."""
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                # Handle grouped QKV parameters
                if g.size(0) == 3 * g.size(1):
                    g = torch.cat(
                        [
                            self.zeropower_fn(g1, steps=group["ns_steps"])
                            for g1 in g.split(g.size(1))
                        ]
                    )
                    scale = g.size(1) ** 0.5
                else:
                    g = self.zeropower_fn(g, steps=group["ns_steps"])
                    scale = max(g.size(0), g.size(1)) ** 0.5
                # Weight decay (decoupled)
                if wd != 0:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(g, alpha=-lr * scale)
