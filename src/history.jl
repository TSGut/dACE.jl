# History: which past velocities each part of a model sees. Environments carry past velocities only (frames up to
# the current time), so a past acceleration is a difference of velocities and never shares a state with a
# finite-difference target ahead of t.

"""
    History(; selfterm = nothing, interactions = (;), use = :velocity,
            selfterm_degree = (8, 4), selfterm_maxm = 2, interaction_degree = 1)

The past states each part of a model sees, in sampling steps (one step: the environments' `dt`, e.g. `K` video
frames; `1:4` means 1 to 4 steps ago). Anything left out sees no past; the current velocities of the agent and
its neighbours are always used. Passed to a basis as `history`.

- `selfterm = steps`: the self term, the agent's own past velocities at `steps`, in a temporal body-order
  expansion in the frame of the current heading (functions of one past state, total degree
  `≤ selfterm_degree[1]`, and products over two, `≤ selfterm_degree[2]`; angular index `≤ selfterm_maxm`). It is
  part of the 1-body term and acts whether or not there are neighbours.
- `interactions = (own = 1, neighbours = 1)`: the reactions to neighbours (the 2-body and higher terms) also
  see the agent's own state one step ago (`own`: the reaction may depend on what the agent was just doing), and
  each neighbour's state one step ago (`neighbours`), polynomials of degree `≤ interaction_degree`. One step only.
- `use = :velocity` (default), `:acceleration` (`(v(t - (k - 1)Δ) - v(t - kΔ)) / Δ`, with `v(t)` the current
  velocity) or `:both`: what a past state is. Velocities are scaled by the basis `vscale`, accelerations by the
  basis `ascale`. At one step, given the current velocity, a past velocity and acceleration carry the same
  information, so in `interactions` `:both` means the acceleration.

Fill the environments with `trajectory_data(...; history)` or `environments(...; history, past_velocities, dt)`.
"""
struct History
    selfterm::Union{Nothing, UnitRange{Int}}
    own::Union{Nothing, UnitRange{Int}}
    neighbours::Union{Nothing, UnitRange{Int}}
    use::Symbol
    selfterm_degree::NTuple{2, Int}
    selfterm_maxm::Int
    interaction_degree::Int
end

steps(x::Integer) = x:x

steps(x) = UnitRange{Int}(x)

function History(; selfterm = nothing, interactions = (;), use = :velocity, selfterm_degree = (8, 4), selfterm_maxm = 2, interaction_degree = 1)
    interactions = isnothing(interactions) ? (;) : interactions
    interactions isa NamedTuple && issubset(keys(interactions), (:own, :neighbours)) ||
        throw(ArgumentError("interactions must be a named tuple with keys own and/or neighbours, e.g. (own = 1,)"))
    selfterm = isnothing(selfterm) ? nothing : steps(selfterm)
    own, nb = (haskey(interactions, k) && !isnothing(interactions[k]) ? steps(interactions[k]) : nothing for k in (:own, :neighbours))
    isnothing(selfterm) && isnothing(own) && isnothing(nb) && throw(ArgumentError("History needs selfterm or interactions"))
    isnothing(selfterm) || (!isempty(selfterm) && first(selfterm) >= 1) || throw(ArgumentError("selfterm must be a non-empty range of steps ≥ 1"))
    for (name, x) in (("own", own), ("neighbours", nb))
        isnothing(x) || x == 1:1 || throw(ArgumentError("interactions: $name supports one step only (1)"))
    end
    use in (:velocity, :acceleration, :both) || throw(ArgumentError("use must be :velocity, :acceleration or :both"))
    length(selfterm_degree) == 2 && all(>=(0), selfterm_degree) || throw(ArgumentError("selfterm_degree must be two non-negative degrees"))
    selfterm_maxm >= 0 && interaction_degree >= 1 || throw(ArgumentError("selfterm_maxm must be non-negative and interaction_degree positive"))
    return History(selfterm, own, nb, use, (Int(selfterm_degree[1]), Int(selfterm_degree[2])), Int(selfterm_maxm), Int(interaction_degree))
end

"""
    history_length(history)

Number of past steps the environments must carry (0 without history).
"""
history_length(::Nothing) = 0

history_length(m::History) = maximum(x -> isnothing(x) ? 0 : last(x), (m.selfterm, m.own, m.neighbours))

has_own_past(m) = !isnothing(m) && !isnothing(m.own)

has_neighbour_past(m) = !isnothing(m) && !isnothing(m.neighbours)

kind(m::History) = m.use === :both ? :acceleration : m.use   # the representation at one step (interactions)

# scale of the one-step past state seen by the interactions of `basis`
past_scale(basis) = kind(basis.history) === :velocity ? basis.vscale : basis.ascale

function check_history(env, H)
    length(env.past_v) >= H || throw(ArgumentError("the basis uses $H past steps but the environment carries $(length(env.past_v)) (see trajectory_data(...; history))"))
    return nothing
end

own_velocity(env, k) = k == 0 ? env.v : env.past_v[k]

# the agent's past state at lag k (k ≥ 1)
own_past(env, k, kind) = kind === :velocity ? env.past_v[k] : (positive_dt(env.dt); (own_velocity(env, k - 1) - env.past_v[k]) / env.dt)

# neighbour j one step earlier: its acceleration, or its velocity relative to the agent's
function neighbour_past(env, j, kind)
    isempty(env.past_vn) && throw(ArgumentError("the basis uses the neighbours' past states but the environment carries none (see trajectory_data(...; history))"))
    return kind === :velocity ? env.past_vn[j] - env.past_v[1] : (positive_dt(env.dt); (env.v + env.u[j] - env.past_vn[j]) / env.dt)
end
