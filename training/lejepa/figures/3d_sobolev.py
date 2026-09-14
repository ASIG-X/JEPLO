import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import sph_harm

# Parameters
n_phi = 400
n_theta = 700
radius = 1.0
alphas = [1, 10, 100]  # More extreme range for contrast
phi = np.linspace(0, np.pi, n_phi)
theta = np.linspace(0, 2 * np.pi, n_theta)
phi, theta = np.meshgrid(phi, theta)
x = radius * np.sin(phi) * np.cos(theta)
y = radius * np.sin(phi) * np.sin(theta)
z = radius * np.cos(phi)


def random_spherical_harmonic_density(phi, theta, max_degree):
    density = np.zeros_like(phi)
    for l in range(max_degree + 1):
        for m in range(-l, l + 1):
            coeff = np.random.randn() + 1j * np.random.randn()
            # No Sobolev weighting here: sharper difference
            density += np.real(coeff * sph_harm(m, l, theta, phi))
    # Nonlinear transformation for contrast
    density = np.abs(density)
    density = np.log1p(density)  # log scale for more contrast
    density -= density.min()
    density /= density.max()
    return density


# Map alpha to max_degree more aggressively
def alpha_to_degree(alpha):
    # Lower alpha = higher degree (more chaos), higher alpha = lower degree (smoother)
    return int(np.clip(60 / (alpha**0.7), 2, 60))


for i, alpha in enumerate(alphas):
    max_degree = alpha_to_degree(alpha)
    density = random_spherical_harmonic_density(phi, theta, max_degree)
    fig = plt.figure(figsize=(5, 5.3))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    colors = cm.inferno(density)
    surf = ax.plot_surface(
        x,
        y,
        z,
        facecolors=colors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=True,
    )
    ax.set_title(r"Sobolev $\alpha$=" + str(alpha), fontsize=28)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.set_facecolor("white")
    ax.set_xlim([-0.65, 0.65])
    ax.set_ylim([-0.65, 0.65])
    ax.set_zlim([-0.65, 0.65])
    ax.dist = 1
    plt.subplots_adjust(0, 0, 1, 0.94)
    plt.savefig(f"3d_sphere_{i}.png", dpi=300)
    plt.close()
