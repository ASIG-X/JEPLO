import numpy as np
import matplotlib.pyplot as plt
from tqdm.rich import trange
import torch
import lejepa as ds


def get_X(N, dim):
    torch.manual_seed(0)
    X = torch.randn(N, dim, device=device)
    X[:, 1] = X[:, 0] * ((torch.rand(X.size(0), device=device) > 0.5).float() * 2 - 1)
    X[:, :2] = X[:, :2] + 1
    return X


N = 100
device = "cpu"
tests = [
    ds.univariate.VCReg(),
    ds.univariate.ExtendedJarqueBera(),
    ds.univariate.CramerVonMises(),
    ds.univariate.Watson(),
    ds.univariate.AndersonDarling(),
    ds.univariate.EppsPulley(),
]
for num_slices in [10, 100]:
    for i, dim in enumerate([128, 1024]):
        fig, axs = plt.subplots(
            2,
            len(tests) + 1,
            sharex="row",
            sharey="row",
            figsize=(2 * (len(tests) + 1), 6),
        )
        X = get_X(N, dim)
        with torch.no_grad():
            axs[0, 0].scatter(
                X[:, 0].cpu(),
                X[:, 1].cpu(),
                c="grey",
                linewidth=1,
                edgecolor="k",
                alpha=0.5,
                zorder=1000,
            )
            axs[1, 0].scatter(
                X[:, 2].cpu(),
                X[:, 3].cpu(),
                c="grey",
                linewidth=1,
                edgecolor="k",
                alpha=0.5,
                zorder=1000,
            )
        axs[0, 0].set_title("original data")
        for j, test in enumerate(tests):
            print(test)
            torch.manual_seed(0)
            if isinstance(test, ds.univariate.UnivariateTest):
                g_loss = ds.multivariate.SlicingUnivariateTest(
                    dim=1, univariate_test=test, num_slices=num_slices
                )
            else:
                g_loss = test
            Xp = X.clone().detach().requires_grad_(True)
            optim = torch.optim.Adam([Xp], lr=0.1)
            losses = []
            for step in trange(1520):
                optim.zero_grad()
                loss = g_loss(Xp)
                losses.append(loss.item())
                loss.backward()
                optim.step()
            # axs[0,j].plot(losses, label=stat.__name__)
            with torch.no_grad():
                axs[0, j + 1].scatter(
                    Xp[:, 0].cpu(),
                    Xp[:, 1].cpu(),
                    c="green",
                    linewidth=1,
                    edgecolor="k",
                    alpha=0.5,
                    zorder=1000,
                )
                axs[1, j + 1].scatter(
                    Xp[:, 2].cpu(),
                    Xp[:, 3].cpu(),
                    c="green",
                    linewidth=1,
                    edgecolor="k",
                    alpha=0.5,
                    zorder=1000,
                )
            axs[0, j + 1].set_title(str(test)[:-2])
        for i in range(axs.shape[1]):
            axs[0, i].set_xlabel("dim 1")
            axs[1, i].set_xlabel("dim 3")
        axs[0, 0].set_ylabel("dim 2")
        axs[1, 0].set_ylabel("dim 4")
        plt.tight_layout()
        plt.savefig(f"2d_slicing_dim_{dim}_N_{N}_slices_{num_slices}.pdf")
        plt.close()
