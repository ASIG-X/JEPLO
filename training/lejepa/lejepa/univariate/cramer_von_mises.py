import torch
from .base import UnivariateTest


class CramerVonMises(UnivariateTest):
    """
    Cramér-von Mises goodness-of-fit test for univariate distributions.

    The Cramér-von Mises test is a non-parametric test that measures the
    discrepancy between the empirical distribution function of a sample and
    the cumulative distribution function of a reference distribution. It is
    generally more powerful than the Kolmogorov-Smirnov test, especially
    for detecting differences in the tails of distributions.

    The test statistic is computed as:
        T = (1/n) * Σ[F(x_i) - (2i-1)/(2n)]²

    where:
        - n is the sample size
        - x_i are the sorted sample values
        - F is the CDF of the reference distribution
        - i is the rank of the observation

    Parameters
    ----------
    x : torch.Tensor
        Input samples to test. Should be shape (n,) for single samples or
        (n, ...) for batched samples. The test is applied along dimension 0.

    Returns
    -------
    T : torch.Tensor
        The test statistic. Lower values indicate better fit to the reference
        distribution. Shape matches the batch dimensions of the input.

    Notes
    -----
    - The input data is assumed to be pre-sorted by `prepare_data()`
    - Critical values depend on the sample size and desired significance level
    - For large samples, asymptotic distributions can be used for p-value calculation

    References
    ----------
    .. [1] Cramér, H. (1928). "On the composition of elementary errors".
           Scandinavian Actuarial Journal, 1928(1), 13-74.
    .. [2] Anderson, T. W. (1962). "On the distribution of the two-sample
           Cramér-von Mises criterion". The Annals of Mathematical Statistics,
           33(3), 1148-1159.
    .. [3] https://en.wikipedia.org/wiki/Cramér–von_Mises_criterion

    Examples
    --------
    >>> # Test if samples follow a standard normal distribution
    >>> x = torch.randn(1000)
    >>> test = CramerVonMises(reference_dist=Normal(0, 1))
    >>> statistic = test(x)
    """

    def forward(self, x):
        s = self.prepare_data(x)
        with torch.no_grad():
            n = x.size(0)
            k = (
                torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
                .mul_(2)
                .sub_(1)
                .div_(2 * n)
            )
            k = k.view(n, *tuple([1] * (x.ndim - 1)))
        T = (k - self.g.cdf(s)).square().mean(0)
        return T
