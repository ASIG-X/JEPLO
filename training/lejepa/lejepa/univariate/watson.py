from .cramer_von_mises import CramerVonMises


class Watson(CramerVonMises):
    """
    Watson's U² test for goodness-of-fit to standard normal N(0,1).
    The Watson test is a location-adjusted modification of the Cramér-von Mises
    test that reduces sensitivity to shifts in location while maintaining power
    to detect differences in scale and shape. It's particularly useful when
    testing for normality when the location parameter is not of primary interest.
    The test statistic is computed as:
        U² = T - (m̄ - 0.5)²
    where:
        - T is the Cramér-von Mises statistic
        - m̄ = (1/n)∑F(x_i) is the mean of the empirical CDF values
        - F is the CDF of the reference distribution N(0,1)
        - The correction term (m̄ - 0.5)² adjusts for location shifts
    For data from N(0,1), m̄ ≈ 0.5, making the correction term small. For
    shifted distributions N(μ,1) with μ ≠ 0, the correction term removes
    the location-based contribution to the CVM statistic.
    Parameters
    ----------
    x : torch.Tensor
        Input samples to test for N(0,1). Should be shape (n, d) where the
        test is applied along dimension 0 for each of d features independently.
    Returns
    -------
    U2 : torch.Tensor
        The Watson U² test statistic. Shape matches the feature dimensions (d,).
        Lower values indicate better fit to N(0,1). Higher values indicate
        departure from standard normality.
    Notes
    -----
    - The Watson statistic satisfies U² ≤ T (CVM statistic) since the
      correction term (m̄ - 0.5)² ≥ 0
    - This test is less powerful than CVM for detecting pure location shifts
      but maintains similar power for detecting scale and shape differences
    - The input data is standardized and sorted by `prepare_data()`
    - Critical values are smaller than those for the CVM test due to the
      location adjustment
    - For large samples from N(0,1), U² follows an approximate distribution
      that can be used for significance testing
    Relationship to CVM
    -------------------
    The Watson test can be written as:
        U² = (1/n)∑[(F(x_(i)) - p_i)² - (m̄ - 0.5)²]
    where p_i = (2i-1)/(2n) are plotting positions. The correction effectively
    centers the empirical CDF before comparing to the theoretical CDF.
    References
    ----------
    .. [1] Watson, G. S. (1961). "Goodness-of-fit tests on a circle".
           Biometrika, 48(1/2), 109-114.
    .. [2] Stephens, M. A. (1974). "EDF Statistics for Goodness of Fit and
           Some Comparisons". Journal of the American Statistical Association,
           69(347), 730-737.
    .. [3] https://en.wikipedia.org/wiki/Cramér–von_Mises_criterion
    See Also
    --------
    CramerVonMises : The base Cramér-von Mises test without location adjustment
    Examples
    --------
    >>> # Test if samples follow N(0,1)
    >>> x = torch.randn(1000, 5)
    >>> test = Watson()
    >>> statistic = test(x)
    >>> # Low values indicate good fit to N(0,1)

    >>> # Watson is less sensitive to location shifts
    >>> x_shifted = torch.randn(1000, 5) + 2.0  # N(2,1)
    >>> stat_shifted = test(x_shifted)
    >>> # stat_shifted will be lower than CVM statistic for same data

    >>> # But still detects scale differences
    >>> x_scaled = torch.randn(1000, 5) * 3.0  # N(0,9)
    >>> stat_scaled = test(x_scaled)
    >>> # stat_scaled will be high, similar to CVM
    """

    def forward(self, x):
        T = super().forward(x)
        m = self.g.cdf(x).mean(0)
        return T - (m - 0.5).square()
