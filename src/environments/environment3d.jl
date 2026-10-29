# 3D: bases in a frame, the heading frame e₃ = v̂ᵢ or the fixed vertical frame e₃ = ẑ; rotations about e₃ by
# the 2D charge rule (at rest, e₃ is the direction of the latest non-zero past velocity, `heading_vector`)

"""
    Environment3(v, r, u; past_v = [], past_vn = [], dt = NaN)

`v = vᵢ`; `r[j] = xⱼ - xᵢ`, `u[j] = vⱼ - vᵢ` for neighbours within the cutoff (3D); `past_v`,
`past_vn`, `dt`: past velocities of the agent and (one step earlier) of each neighbour, as for
`Environment`. A neighbour at `r[j] = 0` is an error, as in 2D.
"""
struct Environment3{T}
    v::SVector{3, T}
    r::Vector{SVector{3, T}}
    u::Vector{SVector{3, T}}
    past_v::Vector{SVector{3, T}}
    past_vn::Vector{SVector{3, T}}
    dt::T
    function Environment3{T}(v, r, u, past_v, past_vn, dt) where {T}
        check_environment(r, u, past_vn)
        return new{T}(v, r, u, past_v, past_vn, dt)
    end
end

Environment3(v::SVector{3, T}, r, u; past_v = SVector{3, T}[], past_vn = SVector{3, T}[], dt = T(NaN)) where {T} =
    Environment3{T}(v, r, u, past_v, past_vn, dt)

Base.length(env::Environment3) = length(env.r)

"""
    rotate(env::Environment3, Q)

Joint rotation by `Q`.
"""
rotate(env::Environment3, Q::AbstractMatrix) = rotate(env, SMatrix{3, 3, Float64}(Q))

rotate(env::Environment3, Q::SMatrix{3, 3}) =
    Environment3(Q * env.v, [Q * r for r in env.r], [Q * u for u in env.u];
        past_v = [Q * w for w in env.past_v], past_vn = [Q * w for w in env.past_vn], env.dt)

"""
    reflect(env::Environment3)

Reflection `(x, y, z) ↦ (x, -y, z)` of all positions and (current and past) velocities.
"""
function reflect(env::Environment3)
    σ(x) = SVector(x[1], -x[2], x[3])
    return Environment3(σ(env.v), σ.(env.r), σ.(env.u); past_v = σ.(env.past_v), past_vn = σ.(env.past_vn), env.dt)
end

# The heading of a 3D agent: its velocity or, at rest, its latest non-zero past velocity (an animal that stops keeps
# facing the way it moved); `nothing` if it has not moved in any stored step.
function heading_vector(env::Environment3)
    norm(env.v) > 0 && return env.v
    for w in env.past_v
        norm(w) > 0 && return w
    end
    return nothing
end

"""
    heading_frame(v) -> (e₁, e₂, e₃)

Right-handed orthonormal frame, `e₃ = v / |v|`.
"""
function heading_frame(v::SVector{3})
    s = norm(v)
    s > 0 || throw(DomainError(v, "the heading frame is undefined for an agent at rest"))
    e3 = v / s
    a = abs(e3[1]) < 0.9 ? SVector(1.0, 0.0, 0.0) : SVector(0.0, 1.0, 0.0)
    e1 = normalize(a - dot(a, e3) * e3)
    return e1, cross(e3, e1), e3
end
