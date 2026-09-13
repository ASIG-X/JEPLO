import torch
from typing import Iterable, Union
import torch


class NLL(torch.nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = torch.distributions.normal.Normal(0, 1)

    def forward(self, x):
        assert x.ndim == 2
        s = x.sort(descending=False, dim=0)[0]
        with torch.no_grad():
            self._k = torch.arange(1, x.size(0) + 1, device=x.device, dtype=torch.float)
            top = self._k.log().sum()
            bottom1 = torch.arange(
                0, x.size(0), device=x.device, dtype=torch.float
            ).log_()
            bottom1[0] = 0
            torch.cumsum(bottom1, dim=0, out=bottom1)
            bottom2 = bottom1.flip(0)
            self._cst = bottom1.neg_().add_(top).sub_(bottom2).mean()
            self._k_m_one = self._k.div_(x.size(0)) - 1 / x.size(0)
            self._N_m_k = 1 - self._k
        Fu = self.g.cdf(s)
        one_m_Fu = (1 - Fu).clip_(self.eps)
        Fu.clip_(self.eps)
        stat = -(
            self._cst
            + Fu.log_().T.matmul(self._k_m_one)
            + one_m_Fu.log_().T.matmul(self._N_m_k)
            + self.g.log_prob(s).mean(0)
        )
        return stat


class SlicingUnivariateTest(torch.nn.Module):
    def __init__(
        self,
        dim: Union[int, Iterable],
        univariate_test,
        num_slices: int,
        reduction: str = "mean",
        sampler: torch.distributions.Distribution = torch.distributions.Normal(
            torch.tensor([0.0]), torch.tensor([1.0])
        ),
    ):
        super().__init__()
        if type(dim) is int:
            dim = [dim]
        self.dim = dim
        self.reduction = reduction
        self.num_slices = num_slices
        self.sampler = sampler
        self.univariate_test = univariate_test

    def forward(self, x):
        cut_dims = list(range(len(self.dim)))
        with torch.no_grad():
            cuts_shape = [x.shape[i] for i in self.dim] + [self.num_slices]
            cuts = self.sampler.sample(sample_shape=cuts_shape).to(
                dtype=x.dtype, device=x.device, non_blocking=True
            )
            cuts = cuts.view(cuts_shape)  # needed as sampler gives an extra dim
            torch.nn.functional.normalize(cuts, p=2, dim=cut_dims, out=cuts)
        # this is now (*, num_slices)
        projected = torch.tensordot(x, cuts, dims=[self.dim, cut_dims])
        projected = projected.flatten(0, -2)  # this is now (num_samples, num_slices)
        assert projected.shape == (x.size(0), self.num_slices)
        stats = self.univariate_test(projected)
        if self.reduction == "mean":
            return stats.mean()
        elif self.reduction == "sum":
            return stats.sum()
        elif self.reduction is None:
            return stats


if __name__ == "__main__":
from datasets import load_dataset
import matplotlib.pyplot as plt

dataset = iter(load_dataset("poloclub/diffusiondb", split="train", trust_remote_code=True,streaming=True).sort("image_nsfw"))
fig, axs = plt.subplots(6,6,figsize=(15,15))
fig2, axs2 = plt.subplots(6,6,figsize=(15,15))

for ax,ax2 in zip(axs.flatten(),axs2.flatten()):
    print("DONE")
    img = next(dataset)["image"]
    ax.imshow(img.resize((512,512)),interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax2.imshow(img.resize((64,64)),interpolation="nearest")
    ax2.set_xticks([])
    ax2.set_yticks([])
fig.tight_layout()
fig.savefig("high_fake.png")
fig2.tight_layout()
fig2.savefig("low_fake.png")

dataset = iter(load_dataset("ILSVRC/imagenet-1k", split="train", trust_remote_code=True,streaming=True))
fig, axs = plt.subplots(6,6,figsize=(15,15))
fig2, axs2 = plt.subplots(6,6,figsize=(15,15))

for ax,ax2 in zip(axs.flatten(),axs2.flatten()):
    print("DONE")
    img = next(dataset)["image"]
    ax.imshow(img.resize((512,512)),interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax2.imshow(img.resize((64,64)),interpolation="nearest")
    ax2.set_xticks([])    
    ax2.set_yticks([])

fig.tight_layout()
fig.savefig("high_real.png")
fig2.tight_layout()
fig2.savefig("low_real.png")

asdf
dataset = load_dataset("poloclub/diffusiondb", split="train", trust_remote_code=True,streaming=True)
img = next(iter(dataset))["image"]
plt.imshow(img.resize((512,512)))
plt.savefig("real.png")
plt.close()
asdf
    asdf

    uni_test = NLL()
    multi_test = SlicingUnivariateTest(
        dim=(1, 2), univariate_test=uni_test, num_slices=100
    )
    print(multi_test(torch.randn(10, 32, 128)))
