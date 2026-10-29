# Self term: the agent's own history as a temporal body-order expansion, part of the 1-body term of a basis with
# `history = History(; selfterm = steps)`. Each past state (a velocity or an acceleration at one step back: a
# channel) enters in the frame of the current heading, like a neighbour does: functions of one channel (temporal
# order 1) and products over two different channels (temporal order 2).

# P_n(2|ṽ| - 1) ṽ⊥^{m0} ∏ₜ P_qₜ(2|x̃⊥| - 1) x̃⊥^{mₜ} P_jₜ(x̃₃), x̃ the scaled state of channel cₜ (`scaled`, the
# basis speed map) in the frame of the current heading (2D: x̃⊥ = ê* x̃, invariant, output ê^charge; 3D: x̃⊥ the
# component in the plane of the basis frame, m0 + Σ mₜ = charge). The angular factors are powers of the scaled
# states, not of their directions, as for the neighbours, so the functions are continuous and equivariant where a
# state (in 3D its perpendicular part, as in straight-line motion) vanishes. With the linear speed map,
# polynomials of the states are in the span (e.g. a persistence a = ρ a(t - Δ)).
struct SelfFunction
    n::Int
    m0::Int
    terms::Vector{NTuple{4, Int}}   # (channel, q, m, j), distinct channels, q + |m| + j ≥ 1
end

degree(f::SelfFunction) = f.n + abs(f.m0) + sum(t -> t[2] + abs(t[3]) + t[4], f.terms; init = 0)

temporal_order(f::SelfFunction) = length(f.terms)

struct SelfTerm
    dim::Int
    channels::Vector{Tuple{Symbol, Int}}   # (kind, lag)
    scales::Vector{Float64}
    vscale::Float64
    inputmap::Symbol
    funcs::Vector{SelfFunction}
    D::Int
end

Base.length(s::SelfTerm) = length(s.funcs)

self_channels(m::History) = [(kind, k) for k in m.selfterm for kind in (:velocity, :acceleration)
                             if m.use === :both || m.use === kind]

# the self term of a basis of dimension `dim` and charge `charge` (`vertical`: 3D fixed frame), or nothing
function self_term(history, dim, charge, vscale, ascale, inputmap; vertical = false)
    (isnothing(history) || isnothing(history.selfterm)) && return nothing
    M = history.selfterm_maxm
    channels = self_channels(history)
    scales = [kind === :velocity ? Float64(vscale) : Float64(ascale) for (kind, _) in channels]
    js = dim == 3 ? (0:max(history.selfterm_degree...)) : (0:0)
    single(c, D) = [(c, q, m, j) for q in 0:D for m in -M:M for j in js if 1 <= q + abs(m) + j <= D]
    m0s = dim == 3 && vertical ? (-M:M) : (0:0)
    ok(m0, ts) = dim == 2 || m0 + sum(t -> t[3], ts) == charge
    funcs = SelfFunction[]
    D1, D2 = history.selfterm_degree
    for c in eachindex(channels), t in single(c, D1), m0 in m0s, n in 0:(D1 - t[2] - abs(t[3]) - t[4] - abs(m0))
        ok(m0, [t]) && push!(funcs, SelfFunction(n, m0, [t]))
    end
    for c1 in eachindex(channels), c2 in (c1 + 1):length(channels), t1 in single(c1, D2), t2 in single(c2, D2), m0 in m0s
        d = sum(t -> t[2] + abs(t[3]) + t[4], (t1, t2)) + abs(m0)
        for n in 0:(D2 - d)
            ok(m0, [t1, t2]) && push!(funcs, SelfFunction(n, m0, [t1, t2]))
        end
    end
    return SelfTerm(dim, channels, scales, Float64(vscale), inputmap, funcs, max(D1, D2, 1))
end

# Legendre polynomials P₀(x) … Pₙ(x) into `P` (in place)
function legendre_table!(P, x, n)
    P[1] = 1.0
    n >= 1 && (P[2] = x)
    for j in 2:n
        P[j + 1] = ((2j - 1) * x * P[j] - (j - 1) * P[j - 1]) / j
    end
    return P
end
legendre_table(x, n) = legendre_table!(zeros(n + 1), x, n)

# scratch of `self_values!` (part of an EvaluationWorkspace): Legendre tables of the speed and of each channel, and
# each channel's scaled state in the frame of the current heading
struct SelfWorkspace
    P0::Vector{Float64}
    Pk::Matrix{Float64}
    Pj::Matrix{Float64}
    ζ::Vector{ComplexF64}
end
SelfWorkspace(s::SelfTerm) = (n = length(s.channels); SelfWorkspace(zeros(s.D + 1), zeros(s.D + 1, n), zeros(s.D + 1, n), zeros(ComplexF64, n)))
self_workspace(::Nothing) = nothing
self_workspace(s::SelfTerm) = SelfWorkspace(s)

# The direction of the agent's heading (2D, as a unit complex number): that of its velocity or, at rest, of its latest
# non-zero past velocity (an animal that stops keeps facing the way it moved), so the self term stays equivariant;
# 0 if it has not moved in any stored step, and then every self feature of non-zero charge is 0.
function heading_direction(env)
    a = abs(complex(env.v[1], env.v[2]))
    a > 0 && return complex(env.v[1], env.v[2]) / a
    for w in env.past_v
        z = complex(w[1], w[2])
        abs(z) > 0 && return z / abs(z)
    end
    return zero(ComplexF64)
end

# values of the self term at `env` into `out`; `frame`: the basis frame in 3D; `w`: scratch (SelfWorkspace)
function self_values!(out, s::SelfTerm, env, charge, frame = nothing, w = SelfWorkspace(s))
    check_history(env, maximum(last, s.channels))
    legendre_table!(w.P0, 2norm(scaled(env.v, s.vscale, s.inputmap)) - 1, s.D)
    e = s.dim == 2 ? heading_direction(env) : zero(ComplexF64)
    for (c, ((kind, k), sc)) in enumerate(zip(s.channels, s.scales))
        x̃ = scaled(own_past(env, k, kind), sc, s.inputmap)
        if s.dim == 2
            z = complex(x̃[1], x̃[2])
            legendre_table!(view(w.Pk, :, c), 2abs(z) - 1, s.D)
            w.ζ[c] = conj(e) * z
        else
            xp, x3 = in_frame(frame, x̃)
            w.ζ[c] = xp
            legendre_table!(view(w.Pk, :, c), 2abs(xp) - 1, s.D)
            legendre_table!(view(w.Pj, :, c), x3, s.D)
        end
    end
    if s.dim == 2
        lead = e^charge
        for (i, f) in enumerate(s.funcs)
            z = lead * w.P0[f.n + 1]
            for (c, q, m, _) in f.terms
                z *= w.Pk[q + 1, c] * cpow(w.ζ[c], m)
            end
            out[i] = z
        end
        return out
    end
    vp, _ = in_frame(frame, scaled(env.v, s.vscale, s.inputmap))
    for (i, f) in enumerate(s.funcs)
        z = complex(w.P0[f.n + 1]) * cpow(vp, f.m0)
        for (c, q, m, j) in f.terms
            z *= w.Pk[q + 1, c] * cpow(w.ζ[c], m) * w.Pj[j + 1, c]
        end
        out[i] = z
    end
    return out
end

self_prior(s::SelfTerm; p = 4) = [1 + degree(f)^p for f in s.funcs]

# columns of the self term follow the interaction functions; all belong to the 1-body term
nself(basis) = isnothing(basis.selfterm) ? 0 : length(basis.selfterm)

function fill_self!(out, basis, env, frame = nothing, w = self_workspace(basis.selfterm))
    isnothing(basis.selfterm) && return out
    K = length(basis.funcs)
    self_values!(view(out, (K + 1):length(out)), basis.selfterm, env, basis.charge, frame, w)
    return out
end
