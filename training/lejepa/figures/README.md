# Bound constant

To generate figure 1, simply run `python bound_constant.py`. This will generate a `bound_constant.pdf` file.

# Sobolev spheres

To generate figure 2, simply run `python 3d_sobolev.py`. This will generate `3d_sphere_0.png`, `3d_sphere_1.png` and `3d_sphere_2.png`.

# Quantile density and variance

To generate figures 1 and 2, simply run `python quantile_variance.py`. To vary `N` (the sample size), simply edit that variable at the top of the file. The execution will produce files `quantile_pdf_*.pdf` and `quantile_variance_*.pdf` over the list of `N` specified by the user.


# Nonparametric example

To generate figure 3, simply run `python nonparametric_example.py`. This will generate figures in the form of `2d_slicing_dim_*_N_*_slices_*.pdf` for the parameters provided in the script. You can modify the number of samples in the data `N`, the number of projections (slices) and the dimension of the data at the top of the file.