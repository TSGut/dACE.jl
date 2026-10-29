@agent struct SwarmAgent(ContinuousAgent{2, Float64}) end
@agent struct SwarmAgent3(ContinuousAgent{3, Float64}) end

agent_type(::Val{2}) = SwarmAgent
agent_type(::Val{3}) = SwarmAgent3

"""
    SwarmTrajectory

`pos[:, i, t]`, `vel[:, i, t]`: agent `i` at time `(t - 1) dt`.
"""
struct SwarmTrajectory{R <: InteractionRule, D}
    rule::R
    pos::Array{Float64, 3}
    vel::Array{Float64, 3}
    dt::Float64
    extent::NTuple{D, Float64}
    periodic::Bool
end

Base.size(tr::SwarmTrajectory) = size(tr.pos)

function synchronous_step!(model)
    rule = model.rule
    r = interaction_radius(rule)
    ids = sort!(collect(allids(model)))
    newvel = map(ids) do id
        a = model[id]
        neighbours = [
            (get_direction(a.pos, b.pos, model), b.vel)
            for b in nearby_agents(a, model, r; search = :exact)
        ]
        velocity_update(rule, a.vel, neighbours, model.dt)
    end
    all(v -> all(isfinite, v) && norm(v) < 1.0e3, newvel) || error(
        "simulation became unstable at step $(abmtime(model)); reduce dt (explicit Euler " *
            "stiffness grows with the number of interacting neighbours)",
    )
    for (id, v) in zip(ids, newvel)
        a = model[id]
        a.vel = v
        move_agent!(a, model, model.dt)
    end
    return nothing
end

"""
    simulate(rule; n_agents, extent, periodic, dt, nsteps, seed, spread, speeds, heading_spread, init)

`D = length(extent) ∈ {2, 3}`. Initial positions uniform in a centred box of side
`spread`, speeds uniform in `speeds`, headings uniform within `heading_spread` of
a random mean heading (interval in 2D, spherical cap in 3D); or `init = (pos, vel)`
(`D × N` arrays).
"""
function simulate(
        rule::InteractionRule;
        n_agents = 50, extent = (50.0, 50.0), periodic = true,
        dt = 0.1, nsteps = 500, seed = 0,
        spread = extent, speeds = (1.0, 1.0), heading_spread = 2π, init = nothing,
    )
    D = length(extent)
    D in (2, 3) || throw(ArgumentError("only 2D and 3D swarms are supported"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be positive and finite, got $dt"))
    extent = Float64.(Tuple(extent))
    spread = Float64.(spread isa Number ? ntuple(_ -> spread, D) : Tuple(spread))
    space = ContinuousSpace(extent; periodic, spacing = interaction_radius(rule) / 2)
    model = StandardABM(
        agent_type(Val(D)), space;
        model_step! = synchronous_step!, container = Vector,
        properties = (; rule, dt), rng = Xoshiro(seed),
    )
    rng = abmrng(model)
    if isnothing(init)
        heading = initial_headings(rng, Val(D), heading_spread)
        for _ in 1:n_agents
            # draw order fixed for reproducibility
            x = SVector(extent) / 2 + (rand(rng, SVector{D}) .- 0.5) .* SVector(spread)
            h = heading()
            s = speeds[1] + (speeds[2] - speeds[1]) * rand(rng)
            add_agent!(x, model, s * h)
        end
    else
        n_agents = size(init[1], 2)
        for i in 1:n_agents
            add_agent!(SVector{D}(init[1][:, i]), model, SVector{D}(init[2][:, i]))
        end
    end

    pos = zeros(D, n_agents, nsteps + 1)
    vel = zeros(D, n_agents, nsteps + 1)
    record!(t) = for i in 1:n_agents
        pos[:, i, t] .= model[i].pos
        vel[:, i, t] .= model[i].vel
    end
    record!(1)
    for t in 1:nsteps
        step!(model)
        record!(t + 1)
    end
    return SwarmTrajectory(rule, pos, vel, dt, extent, periodic)
end

function initial_headings(rng, ::Val{2}, spread)
    θ0 = 2π * rand(rng)
    return () -> (θ = θ0 + spread * (rand(rng) - 0.5); SVector(cos(θ), sin(θ)))
end

function initial_headings(rng, ::Val{3}, spread)
    unit(z, φ) = SVector(sqrt(1 - z^2) * cos(φ), sqrt(1 - z^2) * sin(φ), z)
    n0 = unit(2rand(rng) - 1, 2π * rand(rng))
    a = abs(n0[1]) < 0.9 ? SVector(1.0, 0.0, 0.0) : SVector(0.0, 1.0, 0.0)
    e1 = normalize(a - dot(a, n0) * n0)
    e2 = cross(n0, e1)
    zmin = cos(min(spread, 2π) / 2)
    return function ()
        h = unit(zmin + (1 - zmin) * rand(rng), 2π * rand(rng))
        return h[1] * e1 + h[2] * e2 + h[3] * n0
    end
end
