# PatchMixtureKriging

[![Build Status](https://github.com/RCCWang/PatchMixtureKriging.jl/workflows/CI/badge.svg)](https://github.com/RCCWang/PatchMixtureKriging.jl/actions)
[![Coverage](https://codecov.io/gh/RCCWang/PatchMixtureKriging.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/RCCWang/PatchMixtureKriging.jl)

This package is under construction.

To install, start Julia REPL, press `]`, and run the command `add PatchMixtureKriging`. You might need a few more packages to run the scripts in the example folder; see the first few lines to see which packages are required for each file.

Example folder:
`IBB1D.jl` is a 1D regression example using the iterated Brownian bridge kernel.

 In the example folder, mixGP.jl: The binary splitting tree partitioning (BSP) scheme from Park et. al's "Patchwork Kriging for Large-scale Gaussian Process Regression" (https://jmlr.org/papers/v19/17-042.html) is used to subdivide the input domain into smaller overlapping regions; see `patchGP_partitioning.jl` for a visualization for a 2D domain.

` mixGP.jl`: using the BSP, a separate local Gaussian process regression is run for each region. The final solution is a input-varying convex combination (i.e. the weights depend on the input) of the local regression results. A write-up of the details is coming soon.

### Future works
- Integrate or auto-convert datat types with TemporalGPs.jl and AbstractGP.jl and KernelFunctions.jl
