# the 2D basis (dACEBasis): one-particle functions, centres, construction and direct evaluation

"""
    OneParticle(n, l, mr, mu, q = 0, k = 0)

`φ = fc(ρ) Pₙ(y(ρ)) Pₗ(2|ũ|² - 1) r̂^{mr} ũ^{mu} |b̃|^{2q} b̃^k`, `ũ`, `b̃`: scaled relative
velocity and past state of the neighbour (see `scaled`, `History`), `z^{-m} := z̄^m`, `fc`: the cutoff times the
inner envelope of the basis, if any (`dACEBasis`); charge `mr + mu + k`.
"""
struct OneParticle
    n::Int
    l::Int
    mr::Int
    mu::Int
    q::Int
    k::Int
end

degree(φ::OneParticle) = φ.n + 2φ.l + abs(φ.mr) + abs(φ.mu) + 2φ.q + abs(φ.k)

charge(φ::OneParticle) = φ.mr + φ.mu + φ.k

"""
    Centre(p, m, q = 0, k = 0)

`ψ = Pₚ(2|ṽ|² - 1) ṽ^m |ã|^{2q} ã^k`, `ṽ`, `ã`: scaled velocity and last past state of the
agent (see `scaled`, `History`); charge `m + k`.
"""
struct Centre
    p::Int
    m::Int
    q::Int
    k::Int
end

Centre(p, m) = Centre(p, m, 0, 0)

degree(ψ::Centre) = 2ψ.p + abs(ψ.m) + 2ψ.q + abs(ψ.k)

charge(ψ::Centre) = ψ.m + ψ.k

# μ ∏_b A_b, A_b = Σⱼ ∏_{k ∈ blocks[b]} φₖ(xⱼ)
struct ProductTerm
    μ::Float64
    blocks::Vector{Int}
end

"""
    BasisFunction

`B = ψ(vᵢ) Σ_{j₁≠⋯≠j_N} ∏ₜ φ_{kₜ}(r_{jₜ}, u_{jₜ})`, `N = length(ks)`, stored as
`Σ μ ∏ A_block` (`terms`).
"""
struct BasisFunction
    centre::Int
    ks::Vector{Int}
    terms::Vector{ProductTerm}
end

correlation_order(b::BasisFunction) = length(b.ks)

struct dACEBasis{TR, TS, TP, TM, TH}
    cutoff::Float64
    rscale::Float64
    vscale::Float64
    ascale::Float64
    inputmap::Symbol
    history::TM       # `History` or `nothing`
    selfterm::TH      # self term (`SelfTerm`) or `nothing`
    inner_envelope::Int
    charge::Int
    normaliser::Union{Nothing, NTuple{2, Float64}}
    radial::TR
    speed::TS
    onep::Vector{OneParticle}
    centres::Vector{Centre}
    blocks::Vector{Vector{Int}}
    funcs::Vector{BasisFunction}
    purification::TP  # `nothing` during construction
end

Base.length(basis::dACEBasis) = length(basis.funcs) + nself(basis)

max_order(basis::dACEBasis) = maximum(correlation_order, basis.funcs; init = 0)

"""
    dACEBasis(; maxdeg = (2, 8, 6), cutoff = 5.0, rscale = cutoff / 2, vscale = 2.0, ascale = 1.0,
              maxl = 2, maxm = 2, inner_envelope = 0, filter = (ψ, φs) -> true, pure = true,
              history = nothing, inputmap = :linear, charge = 1, normaliser = nothing)

Basis of total charge `charge` (1: the acceleration `aₓ + i a_y`; 0: invariants). `maxdeg[n]`: total polynomial
degree of the `n`-body terms, so `length(maxdeg)` is the highest body order (`(4, 6, 4)`: 1-, 2- and 3-body).
`rscale`: length scale of the radial map `y(ρ)` (small values resolve short distances finely). `maxl`, `maxm`
bound the degree in the neighbour's relative speed (`l`) and the angular indices `|mr|, |mu|` of its position
and velocity. `inner_envelope = q`: radial envelope `fc(ρ) ((1 - y(ρ))/2)^q`, vanishing at `ρ = 0` for `q > 0`.
`pure = false`: self-interacting products `ψ ∏ₜ Σⱼ φ_{kₜ}(xⱼ)` on the same index set. `filter(ψ, φs)`: keeps
the functions with centre `ψ` and one-particle factors `φs` for which it is true (`socialforce_basis`). `history`: a `History`
(which past states each part of the model sees; `nothing`: none); its self term follows the interaction
functions as part of the 1-body term. `inputmap`: map from velocities (scale `vscale`) and accelerations (scale
`ascale`) to the scaled inputs of the polynomials; `:tanh` keeps all magnitudes below 1, where the polynomials
are resolved evenly (see `scaled`). `normaliser = (c, p)`: every 2-body and higher function is divided by
`(c + S₀)^p`, `S₀ = Σⱼ fc(ρⱼ)` (fixed crowding normalisation; `p = 1`: averaging over neighbours, `p = 1/2`:
Finnis–Sinclair-like).
"""
function dACEBasis(;
        maxdeg = (2, 8, 6), cutoff = 5.0, rscale = cutoff / 2, vscale = 2.0, ascale = 1.0, maxl = 2, maxm = 2,
        inner_envelope = 0, filter = (ψ, φs) -> true, pure = true,
        history = nothing, inputmap = :linear, charge = 1, normaliser = nothing,
    )
    check_basis_options(; inputmap, history, cutoff, rscale, vscale, ascale)
    Dm = isnothing(history) ? 0 : history.interaction_degree
    normaliser = check_normaliser(normaliser)
    order = length(maxdeg) - 1
    D1 = order >= 1 ? maximum(maxdeg[2:end]) : 0
    nmem = has_neighbour_past(history)
    qns, kns = nmem ? (0:(Dm ÷ 2)) : (0:0), nmem ? (-Dm:Dm) : (0:0)
    onep = [
        OneParticle(n, l, mr, mu, q, k)
        for n in 0:D1 for l in 0:maxl for mr in -maxm:maxm for mu in -maxm:maxm for q in qns for k in kns
        if n + 2l + abs(mr) + abs(mu) + 2q + abs(k) <= D1 && 2q + abs(k) <= Dm
    ]
    sort!(onep; by = degree)
    pmax = maximum(maxdeg) ÷ 2
    D = maximum(maxdeg)
    cmem = has_own_past(history)
    centres = [
        Centre(p, m, q, k) for p in 0:pmax for m in -D:D for q in 0:(cmem ? Dm ÷ 2 : 0) for k in (cmem ? (-Dm:Dm) : (0:0))
        if 2p + abs(m) + 2q + abs(k) <= D && 2q + abs(k) <= Dm
    ]

    blockid = Dict{Vector{Int}, Int}()
    blocks = Vector{Vector{Int}}()
    getblock!(b) = get!(blockid, b) do
        push!(blocks, b)
        length(blocks)
    end
    # self-interacting: singleton partition only
    partitions = [pure ? set_partitions(N) : [[[t] for t in 1:N]] for N in 0:order]

    funcs = BasisFunction[]
    for N in 0:order, ks in sorted_tuples(degree.(onep), N, maxdeg[N + 1])
        q = sum(dACE.charge, onep[ks]; init = 0)
        d = sum(degree, onep[ks]; init = 0)
        cs = findall(centres) do ψ
            # the agent's history enters the 1-body term through the self term only
            dACE.charge(ψ) == charge - q && d + degree(ψ) <= maxdeg[N + 1] && filter(ψ, onep[ks]) &&
                (N >= 1 || (ψ.k == 0 && ψ.q == 0))
        end
        isempty(cs) && continue
        acc = Dict{Vector{Int}, Float64}()
        for p in partitions[N + 1]
            bs = sort!([getblock!(sort(ks[B])) for B in p])
            acc[bs] = get(acc, bs, 0.0) + mobius(p)
        end
        terms = [ProductTerm(μ, bs) for (bs, μ) in acc if μ != 0]
        for c in cs
            push!(funcs, BasisFunction(c, ks, terms))
        end
    end

    nmax = maximum(φ -> φ.n, onep; init = 0)
    lmax = max(maximum(φ -> φ.l, onep; init = 0), pmax)
    args = (
        Float64(cutoff), Float64(rscale), Float64(vscale), Float64(ascale), inputmap, history,
        self_term(history, 2, charge, vscale, ascale, inputmap), Int(inner_envelope), Int(charge), normaliser,
        legendre_basis(max(nmax, 1) + 1), legendre_basis(max(lmax, 1) + 1),
        onep, centres, blocks, funcs,
    )
    return dACEBasis(args..., purification(dACEBasis(args..., nothing)))
end

self_frame(::dACEBasis, env) = nothing

# [0, R] → [1, -1]
radial_transform(basis::dACEBasis, ρ) =
    1 - 2 * (1 - exp(-ρ / basis.rscale)) / (1 - exp(-basis.cutoff / basis.rscale))

envelope(basis::dACEBasis, ρ) = basis.inner_envelope == 0 ? fcut(ρ, basis.cutoff) :
    fcut(ρ, basis.cutoff) * ((1 - radial_transform(basis, ρ)) / 2)^basis.inner_envelope

# scaled relative velocity and past state of neighbour `j`, as complex numbers
function neighbour_scaled(basis::dACEBasis, env::Environment, j)
    u = scaled(env.u[j], basis.vscale, basis.inputmap)
    has_neighbour_past(basis.history) || return complex(u[1], u[2]), zero(ComplexF64)
    b = scaled(neighbour_past(env, j, kind(basis.history)), past_scale(basis), basis.inputmap)
    return complex(u[1], u[2]), complex(b[1], b[2])
end

# the agent's scaled last past state for history centres (zero without)
function own_scaled(basis, env)
    has_own_past(basis.history) || return zero(env.v)
    check_history(env, 1)
    return scaled(own_past(env, 1, kind(basis.history)), past_scale(basis), basis.inputmap)
end

"""
    onep_values(basis, env)

`Φ[j, k] = φₖ(r[j], u[j], b[j])`.
"""
function onep_values(basis::dACEBasis, env::Environment)
    Φ = zeros(ComplexF64, length(env), length(basis.onep))
    for j in 1:length(env)
        r = env.r[j]
        ρ = norm(r)
        R = P4ML.evaluate(basis.radial, radial_transform(basis, ρ)) * envelope(basis, ρ)
        ũ, b̃ = neighbour_scaled(basis, env, j)
        S = P4ML.evaluate(basis.speed, 2abs2(ũ) - 1)
        r̂ = complex(r[1], r[2]) / ρ
        for (k, φ) in enumerate(basis.onep)
            Φ[j, k] = R[φ.n + 1] * S[φ.l + 1] * cpow(r̂, φ.mr) * cpow(ũ, φ.mu) * abs2(b̃)^φ.q * cpow(b̃, φ.k)
        end
    end
    return Φ
end

function centre_values(basis::dACEBasis, env::Environment)
    return centre_values!((centres = Vector{ComplexF64}(undef, length(basis.centres)),
        centre_polys = (zeros(length(basis.speed)),)), basis, env)
end

function centre_values!(work, basis::dACEBasis, env::Environment)
    v, a = scaled(env.v, basis.vscale, basis.inputmap), own_scaled(basis, env)
    ṽ, ã = complex(v[1], v[2]), complex(a[1], a[2])
    S = work.centre_polys[1]
    P4ML.evaluate!(S, basis.speed, 2abs2(ṽ) - 1)
    for (i, ψ) in enumerate(basis.centres)
        work.centres[i] = S[ψ.p + 1] * cpow(ṽ, ψ.m) * abs2(ã)^ψ.q * cpow(ã, ψ.k)
    end
    return work.centres
end

"""
    evaluate(basis, env)

All basis functions at `env` (purified).
"""
evaluate(basis::dACEBasis, env::Environment) =
    isnothing(basis.purification) ? evaluate_direct(basis, env) : evaluate(basis, basis.purification, env)

"""
    evaluate_direct(basis, env)

As `evaluate`, with block sums computed directly over neighbours.
"""
function evaluate_direct(basis::dACEBasis, env::Environment)
    Φ = onep_values(basis, env)
    Ψ = centre_values(basis, env)
    A = zeros(ComplexF64, length(basis.blocks))
    for (b, blk) in enumerate(basis.blocks), j in axes(Φ, 1)
        p = one(ComplexF64)
        for k in blk
            p *= Φ[j, k]
        end
        A[b] += p
    end
    return fill_self!(assemble(basis, A, Ψ, interaction_weight(basis, env)), basis, env)
end

"""
    socialforce_basis(; maxdeg = (4, 8), kwargs...)

`aᵢ = s(|vᵢ|) vᵢ + Σⱼ p(ρᵢⱼ) r̂ᵢⱼ + Σⱼ q(ρᵢⱼ, |uᵢⱼ|) uᵢⱼ` as a restricted dACE basis.
"""
function socialforce_basis(; maxdeg = (4, 8), kwargs...)
    length(maxdeg) == 2 || throw(ArgumentError("a social-force model is pairwise"))
    function sf(ψ, φs)
        isempty(φs) && return true
        φ = only(φs)
        return ψ == Centre(0, 0) && ((φ.mr, φ.mu, φ.l) == (1, 0, 0) || (φ.mr, φ.mu) == (0, 1))
    end
    return dACEBasis(; maxdeg, filter = sf, kwargs...)
end

sees_own_past(basis::dACEBasis, f) = (ψ = basis.centres[f.centre]; ψ.q != 0 || ψ.k != 0)

"""
    smoothness_prior(basis; p = 4)

`γₖ = 1 + deg(ψ)ᵖ + Σₜ deg(φₜ)ᵖ` for `Bₖ = ψ ∏ₜ φₜ` (algebraic prior as in ACEpotentials),
for a `dACEBasis`, a `FrameBasis` or a `CompositeBasis`.
"""
smoothness_prior(basis::dACEBasis; p = 4) =
    [[1 + degree(basis.centres[f.centre])^p + sum(k -> degree(basis.onep[k])^p, f.ks; init = 0) for f in basis.funcs];
     isnothing(basis.selfterm) ? Float64[] : self_prior(basis.selfterm; p)]
