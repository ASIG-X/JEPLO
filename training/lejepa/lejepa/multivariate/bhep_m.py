import torch
from .base import MultivariatetTest


class BHEP_M(MultivariatetTest):
    def __init__(self, dim, beta=10):
        super().__init__(dim=dim)
        assert beta > 2
        self.beta = beta

    def forward(self, x):
        x = self.prepare_data(x)
        _, D = x.shape
        norms = x.square().sum(1)
        pair_sim = 2 * x @ x.T + norms + norms.unsqueeze(1)
        lhs = (
            (1 / self.beta ** (D / 2))
            * torch.exp(pair_sim.div(4 * self.beta)).sum()
            / x.size(0)
        )
        rhs = (2 / (self.beta - 0.5) ** (D / 2)) * torch.exp(
            norms / (4 * self.beta - 2)
        ).sum()
        # cst = N / ((self.beta - 1) ** (D / 2))
        return lhs - rhs  # + cst
