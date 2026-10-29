"""
    dACE

Directional atomic cluster expansion: pure body-ordered, rotation-equivariant
expansion of an agent's acceleration in its own velocity and its neighbours'
relative positions and velocities.
"""
module dACE

using ACEfit
import NeighbourLists
using LinearAlgebra, Random, SparseArrays, StaticArrays
import Polynomials4ML as P4ML
using Polynomials4ML: legendre_basis

export Environment, Environment3, environments, trajectory_data, rotate, reflect
export History, dACEBasis, FrameBasis, heading_bases, vertical_bases, socialforce_basis, CompositeBasis
export evaluate, evaluate!, EvaluationWorkspace, body_orders, max_body_order, smoothness_prior
export dACEModel, FrameModel, fit, fit_path, design_statistics, select_lambda, select_basis, predict, body_order_contributions, rmse, rollout, RolloutDivergence
export ARModel, fit_ar1

include("validation.jl")
include("history.jl")
include("environments/environment2d.jl")
include("environments/environment3d.jl")
include("environments/trajectories.jl")
include("basis/common.jl")
include("basis/basis2d.jl")
include("basis/purification2d.jl")
include("basis/selfterm.jl")
include("basis/basis3d.jl")
include("basis/purification3d.jl")
include("basis/bodyorders.jl")
include("basis/workspaces.jl")
include("basis/composite.jl")
include("fitting/qr.jl")
include("fitting/ridge.jl")
include("fitting/evidence.jl")
include("fitting/stages.jl")
include("fitting/normal_equations.jl")
include("models/model2d.jl")
include("fitting/fit.jl")
include("fitting/ar.jl")
include("models/model3d.jl")
include("fitting/fit3d.jl")
include("models/composite.jl")
include("models/prediction.jl")
include("fitting/selection.jl")

end
