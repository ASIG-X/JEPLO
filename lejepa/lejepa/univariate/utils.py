import numpy as np
import torch

# for the optimized implementation: https://gist.github.com/chausies/011df759f167b17b5278264454fff379


def log_norm_cdf_helper(x):
    """Helper function for asymptotic approximation of log normal CDF in the tails.

    Computes a rational approximation used in the asymptotic expansion of the
    Mill's ratio for the standard normal distribution. This is used to compute
    log(Φ(x)) accurately for |x| >> 0 where Φ is the standard normal CDF.

    The approximation is based on:
        sqrt((1 - 0.344) * x + 0.344 * x² + 5.334)

    For large |x|, this approximates the denominator in the asymptotic series:
        1 - Φ(x) ≈ φ(x) / [x + 1/x + 2/x³ + ...] for x >> 0
    where φ(x) is the standard normal PDF.

    Args:
        x: Input tensor, expected to have |x| > 3 for good accuracy

    Returns:
        Approximation term used in tail probability calculations

    Note:
        - For x < 0, ensure the expression under sqrt() is positive
        - Accuracy improves as |x| increases
        - Constants (0.344, 5.334) are empirical/fitted values

    Warning:
        This function is not meant to be called directly. It's a helper
        for log_norm_cdf() to handle extreme tail probabilities.

    References:
        [Add specific reference if known, e.g., Cody (1969) or similar]
    """
    return ((1 - 0.344) * x + 0.344 * x**2 + 5.334).sqrt()


def norm_cdf(x):
    """Compute the standard normal cumulative distribution function.

    Calculates Φ(x) = P(Z ≤ x) where Z ~ N(0, 1) using the relationship
    between the CDF and the error function:

        Φ(x) = (1 + erf(x / √2)) / 2

    This is the standard approach for computing the normal CDF and is
    accurate across the entire real line.

    Args:
        x: Input tensor of any shape

    Returns:
        Tensor of same shape as x containing Φ(x) values in [0, 1]

    Examples:
        >>> x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        >>> norm_cdf(x)
        tensor([0.0013, 0.1587, 0.5000, 0.8413, 0.9987])

    Note:
        - For x = 0: returns exactly 0.5
        - For x → -∞: approaches 0
        - For x → +∞: approaches 1
        - Uses torch.erf which is numerically stable

    See Also:
        log_norm_cdf: For computing log(Φ(x)) with better numerical stability
        torch.erf: The underlying error function implementation
    """
    return (1 + torch.erf(x / np.sqrt(2))) / 2


def log_norm_cdf(x: torch.Tensor, thresh: float = 3.0) -> torch.Tensor:
    """Compute log of the standard normal CDF in a numerically stable way.

    Uses three regions for numerical stability:
    - Middle region: direct computation via log(erf(...))
    - Left tail (x < -thresh): asymptotic approximation to avoid underflow
    - Right tail (x > thresh): log1p to avoid log(1 - ε) cancellation

    Args:
        x: Input tensor
        thresh: Threshold for switching between regions (default: 3.0)

    Returns:
        log(Φ(x)) where Φ is the standard normal CDF

    References:
        [Add citation for the approximation used]
    """
    out = torch.empty_like(x)

    # Define regions
    left = x < -thresh
    right = x > thresh
    middle = ~(left | right)

    # Middle region: direct computation
    if middle.any():
        out[middle] = norm_cdf(x[middle]).log()

    # Left tail: asymptotic approximation
    if left.any():
        x_left = x[left]
        out[left] = (
            -(x_left**2 + np.log(2 * np.pi)) / 2 - log_norm_cdf_helper(-x_left).log()
        )

    # Right tail: use log1p for stability
    if right.any():
        x_right = x[right]
        out[right] = torch.log1p(
            -(-(x_right**2) / 2).exp()
            / np.sqrt(2 * np.pi)
            / log_norm_cdf_helper(x_right)
        )

    return out
