def test_entropy():
    import torch
    import lejepa as ds

    x = torch.randn(50, 1)
    loss = ds.univariate.Entropy()(x).mean()
    x = torch.rand(50, 1)
    assert loss < ds.univariate.Entropy()(x).mean()


def test_watson():
    import torch
    import lejepa as ds

    x = torch.randn(50, 1)
    loss = ds.univariate.Watson()(x).mean()
    x = torch.rand(50, 1)
    assert loss < ds.univariate.Watson()(x).mean()


def test_shapiro_wilk():
    import torch
    import lejepa as ds

    x = torch.randn(50, 1)
    loss = ds.univariate.ShapiroWilk()(x).mean()
    x = torch.rand(50, 1)
    assert loss < ds.univariate.ShapiroWilk()(x).mean()


def test_cramer_von_mises():
    import torch
    import lejepa as ds

    x = torch.randn(50, 1)
    loss = ds.univariate.CramerVonMises()(x).mean()
    x = torch.rand(50, 1)
    assert loss < ds.univariate.CramerVonMises()(x).mean()
