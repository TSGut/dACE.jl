<p align="center">
  <img src="assets/logo.png" alt="dACE logo" width="240">
</p>

<h1 align="center">dACE.jl</h1>

<p align="center">
  <a href="https://github.com/TSGut/dACE.jl/actions/workflows/ci.yml"><img src="https://github.com/TSGut/dACE.jl/actions/workflows/ci.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/TSGut/dACE.jl"><img src="https://codecov.io/gh/TSGut/dACE.jl/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://tsgut.github.io/dACE.jl/dev/"><img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://doi.org/10.5281/zenodo.23762609"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.23762609.svg" alt="DOI"></a>
</p>

dACE.jl learns many-body interaction rules of collectives from trajectory data
using the directional atomic cluster expansion. Its rotation-equivariant models
separate contributions by body order, making learned interactions interpretable
in 2D and 3D.

## Installation

Requires Julia 1.12 or later.

The package is a companion to the paper and is not yet registered.

```julia
using Pkg
Pkg.add(url="https://github.com/TSGut/dACE.jl")
```

## Examples

For examples, see [the documentation](https://tsgut.github.io/dACE.jl/dev/).

<p align="center">
  <img src="assets/rollout_mixed.gif" alt="Rollouts of learned 2-, 3- and 4-body models (colour) against a 2+3+4-body collective they were learned from (grey)." width="100%">
</p>

## References

The paper detailing the theoretical background of the dACE approach is the primary reference for this package: (TODO)

If you find this package useful, please also cite the software itself through its Zenodo record,
[10.5281/zenodo.23762609](https://doi.org/10.5281/zenodo.23762609), which resolves to the latest release.

## Related Packages

dACE.jl is built on [Polynomials4ML.jl](https://github.com/ACEsuit/Polynomials4ML.jl),
[ACEfit.jl](https://github.com/ACEsuit/ACEfit.jl) and [NeighbourLists.jl](https://github.com/ACEsuit/NeighbourLists.jl)
from [ACEsuit](https://github.com/ACEsuit), with benchmark collectives (in the separate helper lib/dACESwarms) simulated
via [Agents.jl](https://github.com/JuliaDynamics/Agents.jl).
