# task-owned evaluation workspaces and in-place evaluation

"""
    EvaluationWorkspace(basis)

Reusable storage for `evaluate!(out, basis, env, workspace)`. Construct one for each
basis and concurrent task; a workspace must not be shared by simultaneous evaluations.
"""
struct EvaluationWorkspace{N, TS}
    pooled::Vector{ComplexF64}
    blocks::Vector{ComplexF64}
    centres::Vector{ComplexF64}
    values::Vector{ComplexF64}
    fcp::Vector{Float64}
    R::Vector{ComplexF64}
    U::Vector{ComplexF64}
    Bq::Vector{Float64}
    Bk::Vector{ComplexF64}
    Bj::Vector{Float64}
    polys::NTuple{N, Vector{Float64}}
    centre_polys::NTuple{2, Vector{Float64}}
    self::TS   # SelfWorkspace, or nothing without a self term
end

pooled_polys(P::Purification) = (zeros(length(P.radial)), zeros(length(P.speed)))

pooled_polys(P::Purification3) = ntuple(_ -> zeros(length(P.poly)), 4)

centre_poly_length(b::dACEBasis) = length(b.speed)

centre_poly_length(b::FrameBasis) = length(b.poly)

neighbour_axial_degree(::Purification) = 0

neighbour_axial_degree(P::Purification3) = P.Jmax

function EvaluationWorkspace(basis, P = basis.purification)
    return EvaluationWorkspace(
        zeros(ComplexF64, length(P.pooled)), zeros(ComplexF64, length(basis.blocks)),
        zeros(ComplexF64, length(basis.centres)), zeros(ComplexF64, length(basis)),
        zeros(P.pmax), zeros(ComplexF64, 2P.Mmax + 1), zeros(ComplexF64, 2P.μmax + 1),
        zeros(P.Qmax + 1), zeros(ComplexF64, 2P.Kmax + 1), zeros(neighbour_axial_degree(P) + 1),
        pooled_polys(P), ntuple(_ -> zeros(centre_poly_length(basis)), 2), self_workspace(basis.selfterm),
    )
end

"""
    evaluate!(out, basis, env, workspace)

Evaluate into `out` using an `EvaluationWorkspace` constructed for `basis`.
The output and workspace are overwritten; the basis and environment are unchanged.
`evaluate!(workspace, basis, env)` evaluates into the workspace's own output vector and returns it.
"""
function evaluate!(out, basis, env, work::EvaluationWorkspace, P = basis.purification)
    length(out) == length(basis) || throw(DimensionMismatch("output length must match basis"))
    no_frame(basis, env) && return fill!(out, 0)
    pooled_features!(work, basis, P, env)
    mul!(work.blocks, P.L, work.pooled)
    centre_values!(work, basis, env)
    assemble!(out, basis, work.blocks, work.centres, interaction_weight(basis, env))
    return fill_self!(out, basis, env, self_frame(basis, env), work.self)
end

evaluate!(work::EvaluationWorkspace, basis, env) = evaluate!(work.values, basis, env, work)

# Unconjugated contraction: dot(B, c) would change the complex-valued model.
contract(B, c) = sum((B[k] * c[k] for k in eachindex(B, c)); init = zero(ComplexF64))

# Each task owns its scratch storage, independently of thread migration or nested calls: task t of a call uses
# `works[t]` only. `works` (workspace_pool) can be kept across calls on the same basis, e.g. the chunks of a QR.
workspace_pool(basis) = [evaluation_workspaces(basis) for _ in 1:Threads.nthreads()]
function foreach_workspace(f, basis, n; works = nothing)
    n == 0 && return nothing
    ntasks = min(Threads.nthreads(), n)
    isnothing(works) && (works = [evaluation_workspaces(basis) for _ in 1:ntasks])   # not `something`: it would build the pool eagerly
    length(works) >= ntasks || throw(ArgumentError("$(length(works)) workspaces for $ntasks tasks"))
    Threads.@threads for task in 1:ntasks
        work = works[task]
        for i in (1 + (task - 1) * n ÷ ntasks):(task * n ÷ ntasks)
            f(work, i)
        end
    end
    return nothing
end

evaluation_workspaces(basis) = EvaluationWorkspace(basis)

evaluation_workspaces(bases::Tuple) = map(evaluation_workspaces, bases)
