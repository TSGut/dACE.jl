# environments and acceleration targets from trajectories (cell-list neighbour search)

"""
    environments(pos, vel; cutoff, extent = nothing, periodic = false, history = nothing, past_velocities = [], dt = NaN)

Environments of all agents of one snapshot (`Environment` in 2D, `Environment3` in 3D); minimum image in a box
`extent` if `periodic`. For a basis with a `History` (`history = basis.history`): `past_velocities[k]` are the
velocities of all agents `k` sampling steps of `dt` earlier (same layout as `vel`, at least
`history_length(history)` of them); each environment keeps the agent's past velocities that the history uses,
and its neighbours' velocities one step earlier if it uses `interactions = (neighbours = 1,)`.
"""
function environments(pos::AbstractMatrix, vel::AbstractMatrix; kwargs...)
    check_snapshot(pos, vel)
    return size(pos, 1) == 2 ? environments(Environment, Val(2), pos, vel; kwargs...) : environments(Environment3, Val(3), pos, vel; kwargs...)
end

# NeighbourLists 0.5 uses 3D positions. Embed 2D data in a non-periodic plane.
# Use the list only for candidate indices: dACE keeps exactly one minimum image
# per other agent, even when the cutoff exceeds half a periodic box length.
function environment_pairlist(X, cutoff, L, ::Val{D}) where {D}
    lo = reduce((a, b) -> min.(a, b), X)
    hi = reduce((a, b) -> max.(a, b), X)
    bound = isnothing(L) ? norm(hi - lo) : norm(L / 2)
    radius = min(Float64(cutoff), bound)
    scale = max(maximum(abs, lo), maximum(abs, hi), bound, radius, 1.0)
    radius += 32eps(scale)  # candidates include roundoff at the strict cutoff
    if isnothing(L)
        lengths = SVector{3, Float64}(ntuple(k -> k <= D ? hi[k] - lo[k] + 2radius : 2radius, 3))
        positions = [SVector{3, Float64}(ntuple(k -> k <= D ? x[k] - lo[k] + radius : radius, 3)) for x in X]
    else
        lengths = SVector{3, Float64}(ntuple(k -> k <= D ? L[k] : 2radius, 3))
        positions = [SVector{3, Float64}(ntuple(k -> k <= D ? mod(x[k], L[k]) : radius, 3)) for x in X]
    end
    pbc = SVector{3, Bool}(ntuple(k -> !isnothing(L) && k <= D, 3))
    return NeighbourLists.PairList(positions, radius, Diagonal(lengths), pbc; int_type = Int, fixcell = false)
end

function environments(E, ::Val{D}, pos, vel; cutoff, extent = nothing, periodic = false, history = nothing, past_velocities = [], dt = NaN) where {D}
    X = [SVector{D, Float64}(ntuple(k -> pos[k, i], D)) for i in axes(pos, 2)]
    V = [SVector{D, Float64}(ntuple(k -> vel[k, i], D)) for i in axes(vel, 2)]
    H = history_length(history)
    length(past_velocities) >= H || throw(ArgumentError("the history uses $H past steps but $(length(past_velocities)) past_velocities are given"))
    H > 0 && positive_dt(dt)
    for (k, P) in enumerate(past_velocities[1:H])
        size(P) == size(vel) || throw(DimensionMismatch("past_velocities[$k] has size $(size(P)), the velocities $(size(vel))"))
    end
    Vp = H == 0 ? nothing : [[SVector{D, Float64}(ntuple(k -> P[k, i], D)) for i in axes(P, 2)] for P in past_velocities[1:H]]
    neighbour_history = has_neighbour_past(history)
    dt = Float64(dt)
    L = periodic ? SVector{D, Float64}(extent) : nothing
    nlist = length(X) > 1 && cutoff > 0 ? environment_pairlist(X, cutoff, L, Val(D)) : nothing
    return map(eachindex(X)) do i
        r = SVector{D, Float64}[]
        u = SVector{D, Float64}[]
        vn = SVector{D, Float64}[]
        # PairList stores neighbours in increasing index order, with periodic images adjacent.
        candidates = isnothing(nlist) ? (1:0) : (nlist.first[i]:(nlist.first[i + 1] - 1))
        previous = 0
        for k in candidates
            j = nlist.j[k]
            (j == i || j == previous) && continue
            previous = j
            dx = X[j] - X[i]
            periodic && (dx = dx - L .* round.(dx ./ L))
            if norm(dx) < cutoff
                push!(r, dx)
                push!(u, V[j] - V[i])
                isnothing(Vp) || isempty(Vp) || !neighbour_history || push!(vn, Vp[1][j])
            end
        end
        E(V[i], r, u; past_v = isnothing(Vp) ? SVector{D, Float64}[] : [P[i] for P in Vp], past_vn = vn, dt)
    end
end

"""
    trajectory_data(pos, vel, dt; cutoff, extent, periodic, times, history = nothing, lagged = false)
        -> (envs, accs) or, with `lagged`, (envs, accs, lagged_envs, lagged_accs)

Environments and `aᵢ = (vᵢ(t+1) - vᵢ(t)) / dt` for the frames `t ∈ times` of `pos`, `vel` (`dt` apart).
`history` (a `History`, e.g. `basis.history`): the environments carry the past velocities that it uses
(frames without enough past are skipped). `lagged`: also the environments and accelerations one frame
earlier, `t - 1` (for the AR(1) diagnostic, `fit_ar1`).
"""
function trajectory_data(
        pos::AbstractArray{<:Any, 3}, vel::AbstractArray{<:Any, 3}, dt;
        cutoff, extent = nothing, periodic = false, times = 1:(size(pos, 3) - 1), history = nothing,
        lagged = false,
    )
    positive_dt(dt)
    check_snapshot(pos, vel)
    all(t -> 1 <= t <= size(pos, 3) - 1, times) ||
        throw(ArgumentError("times must lie in 1:$(size(pos, 3) - 1) (the acceleration at t needs frame t + 1)"))
    D = size(pos, 1)
    E = D == 2 ? Environment{Float64} : Environment3{Float64}
    envs, penvs = E[], E[]
    accs, paccs = SVector{D, Float64}[], SVector{D, Float64}[]
    H = history_length(history)
    snapshot(t) = environments(view(pos, :, :, t), view(vel, :, :, t); cutoff, extent, periodic, history,
        past_velocities = [view(vel, :, :, t - k) for k in 1:H], dt)
    acc(t, i) = SVector{D, Float64}(ntuple(k -> (vel[k, i, t + 1] - vel[k, i, t]) / dt, D))
    for t in times
        t <= H + lagged && continue
        append!(envs, snapshot(t))
        append!(accs, [acc(t, i) for i in axes(vel, 2)])
        if lagged
            append!(penvs, snapshot(t - 1))
            append!(paccs, [acc(t - 1, i) for i in axes(vel, 2)])
        end
    end
    return lagged ? (envs, accs, penvs, paccs) : (envs, accs)
end
