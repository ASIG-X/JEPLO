"""
3D Swiss Roll Manifold with Localized Isotropic Blobs (Coolwarm Colormap)
-------------------------------------------------------------------------

This script generates a high-quality 3D Swiss roll manifold with a density overlay
composed of highly localized, isotropic Gaussian blobs. The density is constructed
in normalized parameter space to ensure circular blobs, and the 'coolwarm' colormap
is used for a lively, eye-catching effect.

Outputs:
    - 'manifold_density_isotropic_blobs.png' (transparent, 300dpi)
    - 'manifold_density_isotropic_blobs.svg' (vector)

Requirements:
    - numpy
    - matplotlib

Usage:
    python manifold_density_isotropic_blobs.py

Author: Metamate AI
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm as gnorm


# --- Manifold: Swiss Roll with Undulations ---
def swiss_roll(u, v):
    """
    Map (u, v) in parameter space to (x, y, z) in 3D Swiss roll with undulations.
    """
    x = u * np.cos(u)
    y = v + 2.0 * np.sin(0.5 * u) * np.cos(0.5 * v)  # undulate in y
    z = u * np.sin(u) + 1.5 * np.cos(0.3 * u + 0.2 * v)  # undulate in z
    return x, y, z


def isotropic_gaussian_2d(u, v, mu, sigma):
    """
    2D isotropic Gaussian PDF.
    """
    diff_u = u - mu[0]
    diff_v = v - mu[1]
    exponent = -0.5 * ((diff_u**2 + diff_v**2) / sigma**2)
    return np.exp(exponent)


def build_isotropic_blobs_density(u, v, K=14, epsilon=0.01):
    """
    Build a density as a GMM with K compact, isotropic, circular blobs in normalized (u,v) space.
    """
    # Normalize u, v to [0,1]
    u_norm = (u - u_min) / (u_max - u_min)
    v_norm = (v - v_min) / (v_max - v_min)
    # Sample means in [0.1, 0.9]^2 to avoid edge effects
    means = np.random.uniform(0.1, 0.9, size=(K, 2))
    # Isotropic sigmas in [0.02, 0.05]
    sigmas = np.random.uniform(0.02, 0.05, K)
    # Positive random weights, normalized
    weights = np.random.uniform(0.7, 1.3, K)
    weights /= weights.sum()
    # Build GMM density
    density = np.zeros_like(u)
    for k in range(K):
        density += weights[k] * isotropic_gaussian_2d(
            u_norm, v_norm, means[k], sigmas[k]
        )
    # Add a tiny baseline to avoid full black background
    density += epsilon
    # Normalize to [0, 1] and apply mild gamma for contrast
    density -= density.min()
    density /= density.max()
    density = np.clip(density, 0, 1)
    density = density**0.98  # mild gamma for punchy but natural look
    return density


# --- Rendering ---
def render_manifold(X, Y, Z, DENS, filename_base="teaser_manifold_0"):
    """
    Render the manifold with density overlay and save as PNG and SVG.
    """
    # Style
    plt.rcParams.update(
        {
            "figure.figsize": (12, 9),
            "font.size": 18,
            "axes.titlesize": 22,
            "axes.labelsize": 18,
            "savefig.dpi": 300,
            "savefig.transparent": True,
            "mathtext.fontset": "stix",
            "axes.edgecolor": "none",
            "axes.facecolor": "none",
            "axes.grid": False,
        }
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Normalize for colormap
    norm = plt.Normalize(vmin=DENS.min(), vmax=DENS.max())
    facecolors = cm.coolwarm(norm(DENS))

    # Surface with explicit facecolors so we truly color by DENS (not Z)
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=1.0,
    )

    # Associate the same norm/cmap to the colorbar via a ScalarMappable
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)
    mappable.set_array(DENS)
    # cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, aspect=30, pad=0.03)
    # cbar.set_label("Density", fontsize=18)
    # cbar.ax.tick_params(labelsize=14)

    # Remove chartjunk
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-12, 12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([np.ptp(X), np.ptp(Y), np.ptp(Z)])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.set_axis_off()

    # Camera view
    ax.view_init(elev=32, azim=128)

    # Save
    plt.subplots_adjust(0, 0, 1, 1)
    plt.savefig(f"{filename_base}.png", transparent=True, dpi=300)
    # plt.savefig(f'{filename_base}.svg', transparent=True)
    plt.show()


"""
2D Euclidean Blobby Density with Custom Arrow Colors, Unit Circle, and Projections
----------------------------------------------------------------------------------

This script visualizes a blobby isotropic GMM density over the square [-2,2] x [-2,2] in Euclidean axes,
draws three colorful arrows (length 1) from the origin (the first is a dark yellow), and overlays a thin
black unit circle. The right subplot shows the projected point densities along each arrow, color-matched,
with the x-axis at the very bottom.

Outputs:
    - 'euclidean_blobs_density_custom.png' (transparent, 300dpi)
    - 'euclidean_blobs_density_custom.svg' (vector)

Requirements:
    - numpy
    - matplotlib

Usage:
    python euclidean_blobs_density_custom.py

Author: Metamate AI
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Reproducibility
np.random.seed(42)

# --- Parameters ---
Nx, Ny = 512, 512  # grid resolution for Euclidean density image
K = 14  # number of Gaussian blobs
sigma_min, sigma_max = 0.04, 0.10
epsilon_baseline = 0.01
n_samples = 20000  # samples used for projection histograms
extent = (-2.0, 2.0, -2.0, 2.0)  # image extent in Cartesian coordinates

# Arrow directions (in radians) and colors
arrow_thetas = np.array([np.deg2rad(25), np.deg2rad(155), np.deg2rad(285)])
arrow_colors = ["#b58900", "tab:orange", "tab:green"]  # dark yellow, orange, green
arrow_labels = ["Arrow A", "Arrow B", "Arrow C"]


# --- Helper functions ---
def isotropic_gaussian_2d(x, y, mu, sigma):
    dx = x - mu[0]
    dy = y - mu[1]
    return np.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))


def sample_gmm_isotropic_in_disk(n, means, sigmas, weights, r_max=2.0):
    K = len(means)
    comp_idx = np.random.choice(K, size=n, p=weights)
    pts = np.zeros((n, 2))
    for k in range(K):
        mask = comp_idx == k
        m = mask.sum()
        if m == 0:
            continue
        pts[mask] = means[k] + sigmas[k] * np.random.randn(m, 2)
    radii = np.linalg.norm(pts, axis=1)
    inside = radii <= r_max
    return pts[inside]


def random_means_in_disk(K, r_min=0.3, r_max=1.6):
    angles = np.random.uniform(0, 2 * np.pi, K)
    radii = np.random.uniform(r_min, r_max, K)
    return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)


# --- Create mixture parameters in extended disk (Cartesian) ---
means = random_means_in_disk(K) * 0.9
means[-1] += 0.4
sigmas = np.random.uniform(sigma_min, sigma_max, K)
weights = np.random.uniform(0.8, 1.2, K)
weights /= weights.sum()

# --- Euclidean grid (cell centers) ---
x_centers = np.linspace(extent[0], extent[1], Nx)
y_centers = np.linspace(extent[2], extent[3], Ny)
Xc, Yc = np.meshgrid(x_centers, y_centers, indexing="xy")  # shapes (Ny, Nx)

# Evaluate density at centers
DENS = np.zeros_like(Xc)
for k in range(K):
    DENS += weights[k] * isotropic_gaussian_2d(Xc, Yc, means[k], sigmas[k])
DENS += epsilon_baseline

# Normalize and mild gamma
DENS_min = DENS.min()
DENS_max = DENS.max()
DENS = (DENS - DENS_min) / (DENS_max - DENS_min)
DENS = DENS**0.5

# Prepare colormap
cmap = cm.coolwarm

# --- Sample points for projection histograms ---
points = sample_gmm_isotropic_in_disk(n_samples, means, sigmas, weights, r_max=2.0)

# --- Projections onto the arrow unit vectors ---
arrow_unit_vecs = np.stack([np.cos(arrow_thetas), np.sin(arrow_thetas)], axis=1)
projections = [points @ uvec for uvec in arrow_unit_vecs]  # each in [-2, 2]

# Smoothed histogram densities with vertical offsets
bins = np.linspace(-4.0, 4.0, 200)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
hist_densities = []
for proj in projections:
    h, _ = np.histogram(proj, bins=bins, density=True)
    # Smooth via simple Gaussian-like kernel convolution
    kernel = np.exp(-0.5 * (np.linspace(-2, 2, 21) ** 2))
    kernel /= kernel.sum()
    h_smooth = np.convolve(h, kernel, mode="same")
    hist_densities.append(h_smooth)

offsets = np.array([0.0, 0.8, 1.6])

# --- Rendering ---
plt.rcParams.update(
    {
        "figure.figsize": (14, 6),
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "savefig.dpi": 300,
        "savefig.transparent": True,
        "mathtext.fontset": "stix",
    }
)

fig = plt.figure(figsize=(12, 6))

# Left subplot: Euclidean density with big-arrow heads, visible axes, and unit circle
ax_cart = fig.add_subplot(1, 2, 1)
norm = plt.Normalize(vmin=DENS.min(), vmax=DENS.max())
img = ax_cart.imshow(
    DENS, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="bilinear"
)
ax_cart.set_aspect("equal", adjustable="box")
# ax_cart.set_title("Euclidean density: localized isotropic blobs", pad=10)

# Draw unit circle (thin black line)
circle = plt.Circle(
    (0, 0), 1.0, edgecolor="black", facecolor="none", linewidth=1.0, zorder=2
)
ax_cart.add_patch(circle)

# Draw colorful arrows from center to unit circle with bigger arrowheads
for th, col, lbl in zip(arrow_thetas, arrow_colors, arrow_labels):
    end = (np.cos(th), np.sin(th))
    ax_cart.annotate(
        "",
        xy=end,
        xytext=(0.0, 0.0),
        arrowprops=dict(
            arrowstyle="-|>", color=col, lw=7.0, mutation_scale=50  # bigger arrowhead
        ),
    )

# Show axes crossing at origin
for spine in ["left", "bottom"]:
    ax_cart.spines[spine].set_position("zero")
for spine in ["top", "right"]:
    ax_cart.spines[spine].set_visible(False)
ax_cart.set_xticks([])
ax_cart.set_yticks([])
ax_cart.tick_params(axis="both", which="both", length=5, size=26)
ax_cart.set_xlim(-1.8, 1.8)
ax_cart.set_ylim(-1.8, 1.8)
ax_cart.set_title("Embedding distribution", fontsize=30)

# Colorbar for density
# cbar = fig.colorbar(img, ax=ax_cart, pad=0.04, shrink=0.85)
# cbar.set_label("Density")

# Right subplot: projected densities along arrows (with vertical offsets)
ax_proj = fig.add_subplot(1, 2, 2)
gauss = gnorm.pdf(bin_centers, loc=0, scale=1)
for dens, col, lbl, off in zip(hist_densities, arrow_colors, arrow_labels, offsets):
    ax_proj.plot(bin_centers, dens + off, color=col, linewidth=2.5, label=None)
    ax_proj.plot(bin_centers, gauss + off, color="k", linewidth=1.5, label=None)
    # ax_proj.fill_between(bin_centers, off, dens + off, color=col, alpha=0.15)
    ax_proj.fill_between(
        bin_centers,
        gauss + off,
        dens + off,
        color="gray",
        alpha=0.3,
        hatch="/",
        label="error" if off == offsets[0] else None,
    )

ax_proj.set_title("Projected point densities", fontsize=30)
ax_proj.set_xlabel("Projection coordinate (along direction)", fontsize=26)

# Keep x-axis at the very bottom and clean up spines/ticks
ax_proj.xaxis.set_ticks_position("bottom")
ax_proj.spines["bottom"].set_position(("outward", 0))
ax_proj.spines["top"].set_visible(False)
ax_proj.spines["right"].set_visible(False)
ax_proj.spines["left"].set_visible(False)
ax_proj.set_yticks([])
ax_proj.set_xticks([-4, -2, 0, 2, 4], labels=[-4, -2, 0, 2, 4], fontsize=20)
ax_proj.set_xlim(-4.0, 4.0)
ymax = offsets[-1] + max(h.max() for h in hist_densities) + 0.2
ax_proj.set_ylim(-0.1, ymax)
ax_proj.legend(fontsize=18)

# ax_proj.legend(loc="upper right", frameon=False)

plt.subplots_adjust(0.0, 0.12, 0.98, 0.92, 0.05, 0.05)
plt.savefig("teaser_manifold_1.png", transparent=True)
# plt.savefig('euclidean_blobs_density_custom.svg', transparent=True)
# plt.show()

# --- Run ---
if __name__ == "__main__":

    # Set reproducibility
    np.random.seed(42)

    # --- Parameters ---
    N_u, N_v = 200, 80  # grid resolution
    u_min, u_max = 1.5 * np.pi, 4.5 * np.pi
    v_min, v_max = -12, 12

    # --- Main Grid ---
    u = np.linspace(u_min, u_max, N_u)
    v = np.linspace(v_min, v_max, N_v)
    U, V = np.meshgrid(u, v, indexing="ij")
    X, Y, Z = swiss_roll(U, V)
    DENS = build_isotropic_blobs_density(U, V, K=14, epsilon=0.01)

    render_manifold(X, Y, Z, DENS)
