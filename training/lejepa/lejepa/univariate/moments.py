import torch
from scipy.stats import norm
from .base import UnivariateTest


class Moments(UnivariateTest):
    def __init__(
        self,
        k_max: int = 4,
        sampler: torch.distributions.Distribution = torch.distributions.Normal(
            torch.tensor([0.0]), torch.tensor([1.0])
        ),
    ):
        super().__init__(sorted=True)
        self.k_max = k_max
        self.sampler = sampler
        moments = []
        for i in range(2, k_max + 1, 2):
            moment_val = norm(loc=0, scale=1).moment(i)
            moments.append(moment_val)
        self.register_buffer(f"moments", torch.Tensor(moments).unsqueeze(1))
        self.register_buffer(f"weights", torch.arange(2, self.k_max + 1).neg().exp())

    def forward(self, x):
        x = self.prepare_data(x)
        k = torch.arange(2, self.k_max + 1, device=x.device, dtype=x.dtype).view(
            -1, 1, 1
        )
        m1 = self.dist_mean(x.mean(0)).abs_()
        if self.k_max >= 2:
            xpow = self.dist_mean((x**k).mean(1))
            xpow[::2].sub_(self.moments)
            m2 = xpow.abs_().T.matmul(self.weights)
            return m1.add_(m2) / self.world_size
        return m1 / self.world_size
