import torch
from .base import UnivariateTest
import numpy as np


class Entropy(UnivariateTest):
    """
    Vasicek, Oldrich (1976). "A Test for Normality Based on Sample Entropy".
    Journal of the Royal Statistical Society. Series B (Methodological).
    """

    def __init__(
        self,
        m: int = 1,
        eps: float = 1e-5,
        method: str = "centered",
        sorted: bool = False,
    ):
        # about 40% of samples are between -0.5 and 0.5
        # so if we observe N samples, int(0.4*N) are supposed to be
        # nearly uniformly distributed between -0.5 and 0.5 hence we should
        # expect an eps of about 1 / (0.4*N)
        super().__init__(eps=eps, sorted=sorted)
        self.m = m
        self.method = method

    def forward(self, x):
        s = self.prepare_data(x)
        cst = np.log(np.sqrt(2 * np.pi * np.exp(1)))
        if self.method == "right":
            stat = s[self.m :].sub(s[: -self.m]).clip(self.eps).log().sum(0) / x.size(0)
            return cst - stat - np.log(x.size(0)) + torch.log(self.m * x.std(0))
        diff = s[self.m * 2 :] - s[: -self.m * 2]
        # with torch.no_grad():
        #     eps = torch.full_like(diff, 1 / (4 * x.size(0)))
        #     eps.sub_(diff).clip_(0)
        stat = diff.clip(self.eps).log().sum(0) / x.size(0)
        for i in range(2 * self.m):
            delta = 1  # (i + 1) / (2 * self.m)
            stat += (s[1 + i] - s[0]).mul(delta).clip(self.eps).log() / x.size(0)
            stat += (s[-1] - s[-2 - i]).mul(delta).clip(self.eps).log() / x.size(0)
        return (
            cst - stat - np.log(x.size(0)) + np.log(2 * self.m) + torch.log(x.std(0))
        ).exp()
