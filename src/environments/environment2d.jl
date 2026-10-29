# the 2D environment of an agent: its velocity and its neighbours' relative positions and velocities

"""
    Environment(v, r, u; past_v = [], past_vn = [], dt = NaN)

`v = vᵢ`; `r[j] = xⱼ - xᵢ`, `u[j] = vⱼ - vᵢ` for neighbours within the cutoff (2D). For bases with
`history` (see `History`): `past_v[k] = vᵢ(t - kΔ)`, the agent's past velocities, `past_vn[j] = vⱼ(t - Δ)`,
each neighbour's velocity one step earlier (empty: none), `dt = Δ`; filled by
`trajectory_data(...; history)`. A neighbour at `r[j] = 0` has no direction and is an error (coincident
positions, e.g. merged tracking identities, have to be removed beforehand).
"""
struct Environment{T}
    v::SVector{2, T}
    r::Vector{SVector{2, T}}
    u::Vector{SVector{2, T}}
    past_v::Vector{SVector{2, T}}
    past_vn::Vector{SVector{2, T}}
    dt::T
    function Environment{T}(v, r, u, past_v, past_vn, dt) where {T}
        check_environment(r, u, past_vn)
        return new{T}(v, r, u, past_v, past_vn, dt)
    end
end

Environment(v::SVector{2, T}, r, u; past_v = SVector{2, T}[], past_vn = SVector{2, T}[], dt = T(NaN)) where {T} =
    Environment{T}(v, r, u, past_v, past_vn, dt)

Base.length(env::Environment) = length(env.r)

"""
    rotate(env, α)

Joint rotation of all positions and (current and past) velocities by `α`.
"""
function rotate(env::Environment, α)
    Q = SMatrix{2, 2}(cos(α), sin(α), -sin(α), cos(α))
    return Environment(Q * env.v, [Q * r for r in env.r], [Q * u for u in env.u];
        past_v = [Q * w for w in env.past_v], past_vn = [Q * w for w in env.past_vn], env.dt)
end

"""
    reflect(env)

Reflection `(x, y) ↦ (x, -y)` of all positions and (current and past) velocities.
"""
function reflect(env::Environment)
    σ(x) = SVector(x[1], -x[2])
    return Environment(σ(env.v), σ.(env.r), σ.(env.u); past_v = σ.(env.past_v), past_vn = σ.(env.past_vn), env.dt)
end
