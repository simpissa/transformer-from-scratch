from collections.abc import Callable, Iterable
import torch
import math
from torch.optim.optimizer import ParamsT
from typing import Optional, Tuple

def CrossEntropyLoss(
    logits: torch.Tensor,
    target: torch.Tensor
):
    logits = logits - logits.max(dim=-1, keepdim=True).values
    denom = torch.log(torch.exp(logits).sum(dim=-1))
    logits = torch.gather(logits, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    loss = torch.mean(denom - logits)
    return loss

class SGD(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3
    ):
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        
    def step(
        self,
        closure: Optional[Callable] = None
    ):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                p.data -= lr * grad
                
        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Optional[Tuple[float, float]] = (0.9, 0.99),
        weight_decay: float = 0.01,
        eps: float = 1e-8
    ):
        beta1, beta2 = betas
        defaults = {"lr": lr, "beta1": beta1, "beta2": beta2, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)
        
    def step(
        self,
        closure: Optional[Callable] = None
    ):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, beta1, beta2, weight_decay, eps = group["lr"], group["beta1"], group["beta2"], group["weight_decay"], group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(grad))
                v = state.get("v", torch.zeros_like(grad))
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                adjusted_lr = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= adjusted_lr * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
                
        return loss

def CosineScheduler(
    t: int,
    lr_max: float,
    lr_min: float,
    T_w: int,
    T_c: int
) -> float: 
    if t < T_w:
        return t / T_w * lr_max
    if T_w <= t <= T_c:
        return lr_min + 1/2 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (lr_max - lr_min)
    return lr_min

def GradientClipping(
    parameters: Iterable[torch.nn.Parameter],
    M: float,
    eps: Optional[float] = 1e-6
):
    l2 = sum(p.grad.pow(2).sum() for p in parameters if p.grad is not None).sqrt()
    
    if l2 >= M:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(M / (l2 + eps))

