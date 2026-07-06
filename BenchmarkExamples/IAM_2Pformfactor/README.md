# Inverse Amplitude Method (IAM)

Inverse amplitude method for unitarizing SU(3) chiral perturbation theory, following the paper:

A. Gomez Nicola, J. R. Pelaez, *Meson-meson scattering within one-loop chiral perturbation theory and its unitarization*, [Phys. Rev. D 65 (2002) 054009](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.65.054009) [[arXiv:hep-ph/0109056](https://arxiv.org/abs/hep-ph/0109056)].

Fit to meson-meson scattering data using coupled-channel IAM with unitarity fulfilled by removing the imaginary part of the loops with unphysical left-hand cuts. The results are reported in 

Y.-J. Shi, C.-Y. Seng, F.-K. Guo, B. Kubis, U.-G. Meißner, W. Wang, *Two-Meson Form Factors in Unitarized Chiral Perturbation Theory*, [arXiv:2011.00921](https://arxiv.org/abs/2011.00921).

The Jupyter notebook `iamfit.ipynb` performing the IAM fit can be run
interactively online — no install, no login — via [Binder](https://mybinder.org):

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fkguo/NativeMinuit.jl/main?urlpath=lab%2Ftree%2FBenchmarkExamples%2FIAM_2Pformfactor%2Fiamfit.ipynb)

The first launch builds the image (several minutes); later launches are cached.
If you land on a **"GitHub API rate limit"** page, that is a transient,
mybinder.org-wide limit (shared token, resets each hour) — just retry a little
later. It is not specific to this repository.

Source repository: [github.com/fkguo/IAMfit](https://github.com/fkguo/IAMfit) — the original IAM fit (notebook + model code) that this benchmark example is adapted from.
