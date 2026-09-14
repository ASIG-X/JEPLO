import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln


def log_C(d, alpha):
    """Compute log of the explicit constant C(d, alpha) to avoid overflow."""
    # log(2^{2*alpha}) = 2*alpha*log(2)
    # log(pi^{d/2}) = (d/2)*log(pi)
    # log(Gamma(alpha + d/2))
    # - log(d)
    # - log(Gamma(alpha))
    # - log(Gamma(d/2))
    return (
        2 * alpha * np.log(2)
        + (d / 2) * np.log(np.pi)
        + gammaln(alpha + d / 2)
        - np.log(d)
        - gammaln(alpha)
        - gammaln(d / 2)
    )


def log_error_bound(d, alpha, M):
    """Compute log of the error bound C(d, alpha) * M^{-2*alpha/d}."""
    return log_C(d, alpha) - (2 * alpha / d) * np.log(M)


# Parameters to explore

d_values = [5]  # d = D-1
alpha_values = np.arange(1, 200)[::20]  # Example smoothness values
M_values = np.arange(1, 200)
cmap = plt.cm.get_cmap("coolwarm")

plt.figure(figsize=(7, 5))
for d in d_values:
    for alpha in alpha_values:
        log_y = log_error_bound(d, alpha, M_values)
        # y = np.exp(log_y)
        label = r"$\alpha$" + f"={alpha}"
        plt.plot(M_values, log_y, label=label, c=cmap(alpha / 200))
# plt.xscale("log")
# plt.yscale("log")
plt.xlim([1, M_values[-1]])
plt.xlabel("Number of directions $M$")
plt.ylabel(r"$C(d, \alpha) \cdot M^{-2\alpha/d}$ (log scale)")
plt.title(f"Decay of Error Bound Constant vs. Number of Directions (D={d})")
plt.legend(ncols=3)
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig("bound_constant.pdf")
plt.close()
