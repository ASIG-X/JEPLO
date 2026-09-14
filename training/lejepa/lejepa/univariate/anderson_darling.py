import torch
from .base import UnivariateTest
from .utils import log_norm_cdf


class AndersonDarling(UnivariateTest):
    """
    Anderson-Darling goodness-of-fit test for normality.

    The Anderson-Darling test is a non-parametric test that determines whether
    a sample comes from a normal distribution. It is a modification of the
    Cramér-von Mises test that gives more weight to observations in the tails
    of the distribution, making it particularly effective at detecting departures
    from normality in the extremes.

    The test statistic is computed as:
        A² = -Σ[w_i * (log(Φ(x_i)) + log(Φ(-x_{n+1-i})))]

    where:
        - n is the sample size
        - x_i are the sorted sample values
        - Φ is the standard normal CDF
        - w_i = (2i-1) / n² are the weights
        - The symmetric term with -x_{n+1-i} exploits normality's symmetry

    Parameters
    ----------
    x : torch.Tensor
        Input samples to test for normality. Should be shape (n,) for single
        samples or (n, ...) for batched samples. The test is applied along
        dimension 0.

    Returns
    -------
    A : torch.Tensor
        The Anderson-Darling test statistic. Higher values indicate stronger
        evidence against normality. Shape matches the batch dimensions of the input.

    Notes
    -----
    - The input data is standardized and sorted by `prepare_data()`
    - This implementation uses logarithmic CDFs for numerical stability
    - Critical values at common significance levels (α):
        * α = 0.15: 1.621
        * α = 0.10: 1.933
        * α = 0.05: 2.492
        * α = 0.025: 3.070
        * α = 0.01: 3.857
    - The test is more powerful than Kolmogorov-Smirnov for detecting
      non-normality, especially in the distribution tails

    References
    ----------
    .. [1] Anderson, T. W. and Darling, D. A. (1952). "Asymptotic theory of
           certain 'goodness of fit' criteria based on stochastic processes".
           The Annals of Mathematical Statistics, 23(2), 193-212.
    .. [2] Stephens, M. A. (1974). "EDF Statistics for Goodness of Fit and
           Some Comparisons". Journal of the American Statistical Association,
           69(347), 730-737.
    .. [3] https://en.wikipedia.org/wiki/Anderson–Darling_test

    Examples
    --------
    >>> # Test if samples follow a normal distribution
    >>> x = torch.randn(1000)
    >>> test = AndersonDarling()
    >>> statistic = test(x)
    >>> # If statistic > 2.492, reject normality at 5% significance level
    """

    def forward(self, x):
        s = self.prepare_data(x)
        n = x.size(0)

        with torch.no_grad():
            k = (
                torch.arange(1, n + 1, device=x.device, dtype=torch.float)
                .mul_(2)
                .sub_(1)
            )

        A = log_norm_cdf(s) + log_norm_cdf(-s.flip(0))
        A_squared = -n - torch.tensordot(A, k, [[0], [0]]) / n

        return A_squared
