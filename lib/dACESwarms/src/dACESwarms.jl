"""
    dACESwarms

Noise-free trajectories of collectives with known rules, updated synchronously:
`vᵢ(t+dt) = f(vᵢ(t), {(xⱼ - xᵢ, vⱼ)})` from one frozen snapshot.
"""
module dACESwarms

using Agents, LinearAlgebra, Random, StaticArrays

export InteractionRule, ForceRule, Boids, DOrsogna, Zonal
export velocity_update, acceleration, interaction_radius
export SwarmTrajectory, simulate

include("rules.jl")
include("simulate.jl")

end
