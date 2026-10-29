# the 3D basis in the heading (or vertical) frame (FrameBasis): one-particle functions, centres, construction

"""
    OneParticle3(n, a, mr, l, b, mu, k = 0, j = 0)

`φ = fc Pₙ(y) P_a(t) w^{mr} P_l(2|ũ⊥|² - 1) P_b(ũ₃) ũ⊥^{mu} b̃⊥^k b̃₃^j`, `r̂ = (w, t)`,
`ũ = (ũ⊥, ũ₃)` and `b̃ = (b̃⊥, b̃₃)` (scaled relative velocity and past state of the
neighbour, see `History`) in the frame; charge `mr + mu + k`.
"""
struct OneParticle3
    n::Int
    a::Int
    mr::Int
    l::Int
    b::Int
    mu::Int
    k::Int
    j::Int
end

OneParticle3(n, a, mr, l, b, mu) = OneParticle3(n, a, mr, l, b, mu, 0, 0)

degree(φ::OneParticle3) = φ.n + φ.a + abs(φ.mr) + 2φ.l + φ.b + abs(φ.mu) + abs(φ.k) + φ.j

charge(φ::OneParticle3) = φ.mr + φ.mu + φ.k

"""
    FrameBasis(; maxdeg = (4, 8, 6), charge = 1, cutoff = 5.0, rscale = cutoff / 2,
               vscale = 2.0, ascale = 1.0, maxm = 1, maxl = 0, maxa = 2, maxb = 1, vertical = false,
               history = nothing, inputmap = :linear, normaliser = nothing)

3D basis in a frame `(e₁, e₂, e₃)`: the heading frame `e₃ = v̂ᵢ` (default), or with `vertical = true` the
fixed frame `e₃ = ẑ` (symmetry: rotations about the vertical only, e.g. under gravity). `charge = 1`: the
part of the acceleration in the plane `a₁ + i a₂`; `charge = 0`: the part along the axis, `a₃` (built in
pairs by `heading_bases` and `vertical_bases`). Centres `Pₚ(2|ṽ| - 1)` (heading frame) or
`Pₚ(2|ṽ⊥|² - 1) P_q(ṽ₃) ṽ⊥^m` of charge `m` (vertical frame).

Degrees, compared with `dACEBasis`: `maxdeg[n]` is the total degree of the `n`-body terms, as in 2D. Per
neighbour, `maxm` bounds the angular indices in the plane (as in 2D), `maxl` the degree in the in-plane
relative speed (as `maxl` in 2D), `maxa` the degree in the axial direction of the neighbour (`t = r̂ ⋅ e₃`),
`maxb` the degree in its axial relative velocity (`ũ₃`), the two directions 2D does not have. `history`, `ascale`,
`inputmap`, `normaliser`: as for `dACEBasis`; past states enter in the frame (history centres `ã⊥^k ã₃^j`,
neighbour factors `b̃⊥^k b̃₃^j`, `|k| + j ≤ interaction_degree`; the self term in the frame of the basis).
"""
struct FrameBasis{TL, TP, TM, TH}
    cutoff::Float64
    rscale::Float64
    vscale::Float64
    ascale::Float64
    poly::TL
    charge::Int
    vertical::Bool
    inputmap::Symbol
    history::TM
    selfterm::TH
    normaliser::Union{Nothing, NTuple{2, Float64}}
    onep::Vector{OneParticle3}
    centres::Vector{NTuple{5, Int}}
    blocks::Vector{Vector{Int}}
    funcs::Vector{BasisFunction}
    purification::TP
end

Base.length(basis::FrameBasis) = length(basis.funcs) + nself(basis)

max_order(basis::FrameBasis) = maximum(correlation_order, basis.funcs; init = 0)

radial_transform(basis::FrameBasis, ρ) =
    1 - 2 * (1 - exp(-ρ / basis.rscale)) / (1 - exp(-basis.cutoff / basis.rscale))

function FrameBasis(;
        maxdeg = (4, 8, 6), charge = 1, cutoff = 5.0, rscale = cutoff / 2, vscale = 2.0, ascale = 1.0,
        maxm = 1, maxl = 0, maxa = 2, maxb = 1, vertical = false,
        history = nothing, inputmap = :linear, normaliser = nothing,
    )
    check_basis_options(; inputmap, history, cutoff, rscale, vscale, ascale)
    Dm = isnothing(history) ? 0 : history.interaction_degree
    normaliser = check_normaliser(normaliser)
    order = length(maxdeg) - 1
    D1 = order >= 1 ? maximum(maxdeg[2:end]) : 0
    nmem = has_neighbour_past(history)
    kns, jns = nmem ? (-Dm:Dm) : (0:0), nmem ? (0:Dm) : (0:0)
    onep = [
        OneParticle3(n, a, mr, l, b, mu, k, j)
        for n in 0:D1 for a in 0:maxa for mr in -maxm:maxm for l in 0:maxl for b in 0:maxb
        for mu in -maxm:maxm for k in kns for j in jns if n + a + abs(mr) + 2l + b + abs(mu) + abs(k) + j <= D1 && abs(k) + j <= Dm
    ]
    sort!(onep; by = degree)
    D = maximum(maxdeg)
    cmem = has_own_past(history)
    ks, js = cmem ? (-Dm:Dm) : (0:0), cmem ? (0:Dm) : (0:0)
    centres = vertical ?
        [(p, q, m, k, j) for p in 0:(D ÷ 2) for q in 0:D for m in -D:D for k in ks for j in js if 2p + q + abs(m) + abs(k) + j <= D && abs(k) + j <= Dm] :
        [(p, 0, 0, k, j) for p in 0:D for k in ks for j in js if p + abs(k) + j <= D && abs(k) + j <= Dm]
    cdeg(c) = centre_degree(vertical, c)

    blockid = Dict{Vector{Int}, Int}()
    blocks = Vector{Vector{Int}}()
    getblock!(b) = get!(blockid, b) do
        push!(blocks, b)
        length(blocks)
    end
    partitions = [set_partitions(N) for N in 0:order]
    funcs = BasisFunction[]
    for N in 0:order, ks in sorted_tuples(degree.(onep), N, maxdeg[N + 1])
        q = sum(dACE.charge, onep[ks]; init = 0)
        d = sum(degree, onep[ks]; init = 0)
        # the agent's history enters the 1-body term through the self term only
        cs = findall(c -> q + c[3] + c[4] == charge && d + cdeg(c) <= maxdeg[N + 1] &&
            (N >= 1 || (c[4] == 0 && c[5] == 0)), centres)
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

    npoly = max(D1, maxa, maxl, maxb, maximum(maxdeg), 1) + 1
    args = (
        Float64(cutoff), Float64(rscale), Float64(vscale), Float64(ascale), legendre_basis(npoly), charge, vertical,
        inputmap, history, self_term(history, 3, charge, vscale, ascale, inputmap; vertical), normaliser, onep, centres, blocks, funcs,
    )
    return FrameBasis(args..., purification(FrameBasis(args..., nothing)))
end

sees_own_past(basis::FrameBasis, f) = (c = basis.centres[f.centre]; c[4] != 0 || c[5] != 0)

const VERTICAL_FRAME = (SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0))

# The frame of a basis at `env`: the vertical frame, or the heading frame of `heading_vector(env)`. An agent that has
# not moved has no heading frame: its heading-frame features are all 0 (`no_frame`), and the vertical frame stands in.
no_frame(::dACEBasis, env) = false
no_frame(basis::FrameBasis, env) = !basis.vertical && isnothing(heading_vector(env))
basis_frame(basis::FrameBasis, env::Environment3) = no_frame(basis, env) ? VERTICAL_FRAME :
    basis.vertical ? VERTICAL_FRAME : heading_frame(heading_vector(env))

self_frame(basis::FrameBasis, env) = isnothing(basis.selfterm) ? nothing : basis_frame(basis, env)

# components (x⊥, x₃) of `x` in the frame
in_frame(frame, x) = (complex(dot(x, frame[1]), dot(x, frame[2])), dot(x, frame[3]))

# (ρ, w, t, ũ⊥, ũ₃, b̃⊥, b̃₃) of neighbour `j`
function frame_coordinates(basis::FrameBasis, frame, env::Environment3, j)
    r = env.r[j]
    ρ = norm(r)
    w, t = in_frame(frame, r / ρ)
    ũp, ũ₃ = in_frame(frame, scaled(env.u[j], basis.vscale, basis.inputmap))
    b̃p, b̃₃ = if has_neighbour_past(basis.history)
        in_frame(frame, scaled(neighbour_past(env, j, kind(basis.history)), past_scale(basis), basis.inputmap))
    else
        zero(ComplexF64), 0.0
    end
    return ρ, w, t, ũp, ũ₃, b̃p, b̃₃
end

function onep_values(basis::FrameBasis, env::Environment3)
    frame = basis_frame(basis, env)
    Φ = zeros(ComplexF64, length(env), length(basis.onep))
    for j in 1:length(env)
        ρ, w, t, ũp, ũ₃, b̃p, b̃₃ = frame_coordinates(basis, frame, env, j)
        Ly = P4ML.evaluate(basis.poly, radial_transform(basis, ρ)) * fcut(ρ, basis.cutoff)
        Lt = P4ML.evaluate(basis.poly, t)
        Lz = P4ML.evaluate(basis.poly, 2abs2(ũp) - 1)
        Lb = P4ML.evaluate(basis.poly, ũ₃)
        for (k, φ) in enumerate(basis.onep)
            Φ[j, k] = Ly[φ.n + 1] * Lt[φ.a + 1] * cpow(w, φ.mr) * Lz[φ.l + 1] * Lb[φ.b + 1] * cpow(ũp, φ.mu) *
                cpow(b̃p, φ.k) * b̃₃^φ.j
        end
    end
    return Φ
end

centre_degree(vertical, c) = (vertical ? 2c[1] + c[2] + abs(c[3]) : c[1]) + abs(c[4]) + c[5]

function centre_values(basis::FrameBasis, env::Environment3)
    return centre_values!((centres = Vector{ComplexF64}(undef, length(basis.centres)),
        centre_polys = ntuple(_ -> zeros(length(basis.poly)), 2)), basis, env)
end

function centre_values!(work, basis::FrameBasis, env::Environment3)
    frame = basis_frame(basis, env)
    ap, a3 = in_frame(frame, own_scaled(basis, env))
    mem(c) = cpow(ap, c[4]) * a3^c[5]
    ṽ = scaled(env.v, basis.vscale, basis.inputmap)
    if !basis.vertical
        L = work.centre_polys[1]
        P4ML.evaluate!(L, basis.poly, 2norm(ṽ) - 1)
        for (i, c) in enumerate(basis.centres)
            work.centres[i] = L[c[1] + 1] * mem(c)
        end
        return work.centres
    end
    vp = complex(ṽ[1], ṽ[2])
    Lp, Lq = work.centre_polys
    P4ML.evaluate!(Lp, basis.poly, 2abs2(vp) - 1)
    P4ML.evaluate!(Lq, basis.poly, ṽ[3])
    for (i, c) in enumerate(basis.centres)
        work.centres[i] = Lp[c[1] + 1] * Lq[c[2] + 1] * cpow(vp, c[3]) * mem(c)
    end
    return work.centres
end

function evaluate_direct(basis::FrameBasis, env::Environment3)
    no_frame(basis, env) && return zeros(ComplexF64, length(basis))
    Φ = onep_values(basis, env)
    A = zeros(ComplexF64, length(basis.blocks))
    for (b, blk) in enumerate(basis.blocks), j in axes(Φ, 1)
        p = one(ComplexF64)
        for k in blk
            p *= Φ[j, k]
        end
        A[b] += p
    end
    return fill_self!(assemble(basis, A, centre_values(basis, env), interaction_weight(basis, env)), basis, env, basis_frame(basis, env))
end

evaluate(basis::FrameBasis, env::Environment3) =
    isnothing(basis.purification) ? evaluate_direct(basis, env) :
    evaluate!(EvaluationWorkspace(basis), basis, env)

"""
    heading_bases(; kwargs...) -> (plane, axis)

The two `FrameBasis` of a 3D model in the heading frame: `plane` (charge 1) for the part of the acceleration
across the heading, `axis` (charge 0) for the part along it.
"""
heading_bases(; kwargs...) = (FrameBasis(; charge = 1, kwargs...), FrameBasis(; charge = 0, kwargs...))

"""
    vertical_bases(; kwargs...) -> (plane, axis)

`heading_bases` in the fixed frame `e₃ = ẑ`: `plane` for the horizontal acceleration `a_x + i a_y`, `axis`
for the vertical `a_z`.
"""
vertical_bases(; kwargs...) = heading_bases(; vertical = true, kwargs...)

smoothness_prior(basis::FrameBasis; p = 4) =
    [[1 + centre_degree(basis.vertical, basis.centres[f.centre])^p + sum(k -> degree(basis.onep[k])^p, f.ks; init = 0) for f in basis.funcs];
     isnothing(basis.selfterm) ? Float64[] : self_prior(basis.selfterm; p)]
