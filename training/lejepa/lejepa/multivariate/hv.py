import torch
from .base import MultivariatetTest


class HV(MultivariatetTest):
    def __init__(self, gamma=1):
        super().__init__()
        assert gamma > 0
        self.gamma = gamma

    def forward(self, x):
        x = self.prepare_data(x)
        N, D = x.shape
        norms = x.square().sum(1)
        pair_sim = 2 * x @ x.T + norms + norms.unsqueeze(1)
        lhs = torch.exp(pair_sim.div(4 * self.gamma))
        rhs = (
            x @ x.T
            - pair_sim / (2 * self.gamma)
            + D / (2 * self.gamma)
            + pair_sim / (4 * self.gamma**2)
        )
        # cst = N/((beta-1)**(D/2))
        return (lhs * rhs).sum() / N
