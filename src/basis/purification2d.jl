# Purification (Ho, Gutleb, Ortner, arXiv:2401.01550), pooled per power p = |β| of fc

"""
    Purification

`A = L * Ã`: block sums from pooled features `Ã`, `pooled[f] = (p, ν, λ, M, μ, Q, K)`.
"""
struct Purification{TR, TV}
    pooled::Vector{NTuple{7, Int}}
    L::SparseMatrixCSC{Float64, Int}
    νmax::Int
    λmax::Int
    Mmax::Int
    μmax::Int
    pmax::Int
    Qmax::Int
    Kmax::Int
    radial::TR
    speed::TV
end

# Legendre coefficients of a polynomial from its values at the nodes of `factor` (the QR of the Vandermonde matrix)
function legendre_coefficients(factor::Factorization, f)
    c = factor \ f
    c[abs.(c) .< 1.0e-13 * max(1.0, maximum(abs, c))] .= 0
    return c
end

function purification(basis::dACEBasis)
    onep = basis.onep
    deg_r(β) = sum(onep[k].n for k in β)
    q_u(β) = (sum(abs(onep[k].mu) for k in β) - abs(sum(onep[k].mu for k in β))) ÷ 2
    deg_v(β) = sum(onep[k].l for k in β) + q_u(β)
    # ∏ₜ |b̃|^{2qₜ} b̃^{kₜ} = |b̃|^{2Q} b̃^K
    K_b(β) = sum(onep[k].k for k in β)
    Q_b(β) = sum(onep[k].q for k in β) + (sum(abs(onep[k].k) for k in β) - abs(K_b(β))) ÷ 2
    νmax = maximum(deg_r, basis.blocks; init = 0)
    λmax = maximum(deg_v, basis.blocks; init = 0)
    Mmax = maximum(β -> abs(sum(onep[k].mr for k in β)), basis.blocks; init = 0)
    μmax = maximum(β -> abs(sum(onep[k].mu for k in β)), basis.blocks; init = 0)
    pmax = maximum(length, basis.blocks; init = 0)
    Qmax = maximum(Q_b, basis.blocks; init = 0)
    Kmax = maximum(β -> abs(K_b(β)), basis.blocks; init = 0)

    Pr, Pv = legendre_basis(max(νmax, 1) + 1), legendre_basis(max(λmax, 1) + 1)
    n = max(νmax, λmax) + 8
    x = [cos(π * (i - 0.5) / n) for i in 1:n]
    Vr = P4ML.evaluate(basis.radial, x)
    Vs = P4ML.evaluate(basis.speed, x)
    Fr = qr(P4ML.evaluate(Pr, x), ColumnNorm())
    Fv = qr(P4ML.evaluate(Pv, x), ColumnNorm())

    index = Dict{NTuple{7, Int}, Int}()
    pooled = NTuple{7, Int}[]
    I, J, W = Int[], Int[], Float64[]
    for (b, β) in enumerate(basis.blocks)
        p = length(β)
        M = sum(onep[k].mr for k in β)
        μ = sum(onep[k].mu for k in β)
        fr = reduce(.*, (Vr[:, onep[k].n + 1] for k in β))
        fv = reduce(.*, (Vs[:, onep[k].l + 1] for k in β)) .* ((x .+ 1) ./ 2) .^ q_u(β)  # |ũ|² = (z + 1)/2
        cr = legendre_coefficients(Fr, fr)
        cv = legendre_coefficients(Fv, fv)
        for ν in findall(!iszero, cr), λ in findall(!iszero, cv)
            key = (p, ν - 1, λ - 1, M, μ, Q_b(β), K_b(β))
            f = get!(index, key) do
                push!(pooled, key)
                length(pooled)
            end
            push!(I, b); push!(J, f); push!(W, cr[ν] * cv[λ])
        end
    end
    L = sparse(I, J, W, length(basis.blocks), length(pooled))
    return Purification(pooled, L, νmax, λmax, Mmax, μmax, pmax, Qmax, Kmax, Pr, Pv)
end

"""
    pooled_features(basis, P, env)

`Ã_{pνλMμQK} = Σⱼ fc^p P_ν(y) P_λ(2|ũ|² - 1) r̂^M ũ^μ |b̃|^{2Q} b̃^K`.
"""
function pooled_features(basis::dACEBasis, P::Purification, env::Environment)
    return pooled_features!(EvaluationWorkspace(basis, P), basis, P, env)
end

function pooled_features!(work, basis::dACEBasis, P::Purification, env::Environment)
    Pr, Pv = P.radial, P.speed
    Ã, fcp, R, U, Bq, Bk = work.pooled, work.fcp, work.R, work.U, work.Bq, work.Bk
    y, z = work.polys
    fill!(Ã, 0)
    for j in 1:length(env)
        r = env.r[j]
        ρ = norm(r)
        fc = envelope(basis, ρ)
        fc == 0 && continue
        for p in 1:P.pmax
            fcp[p] = fc^p
        end
        P4ML.evaluate!(y, Pr, radial_transform(basis, ρ))
        ũ, b̃ = neighbour_scaled(basis, env, j)
        P4ML.evaluate!(z, Pv, 2abs2(ũ) - 1)
        r̂ = complex(r[1], r[2]) / ρ
        for m in -P.Mmax:P.Mmax
            R[m + P.Mmax + 1] = cpow(r̂, m)
        end
        for m in -P.μmax:P.μmax
            U[m + P.μmax + 1] = cpow(ũ, m)
        end
        for q in 0:P.Qmax
            Bq[q + 1] = abs2(b̃)^q
        end
        for k in -P.Kmax:P.Kmax
            Bk[k + P.Kmax + 1] = cpow(b̃, k)
        end
        for (f, (p, ν, λ, M, μ, Q, K)) in enumerate(P.pooled)
            Ã[f] += fcp[p] * y[ν + 1] * z[λ + 1] * R[M + P.Mmax + 1] * U[μ + P.μmax + 1] * Bq[Q + 1] * Bk[K + P.Kmax + 1]
        end
    end
    return Ã
end

# `w` multiplies every function of correlation order ≥ 1 (see `interaction_weight`)
function assemble(basis, A, Ψ, w = 1.0)
    return assemble!(Vector{ComplexF64}(undef, length(basis)), basis, A, Ψ, w)
end

function assemble!(out, basis, A, Ψ, w = 1.0)
    for (i, f) in enumerate(basis.funcs)
        val = isempty(f.terms) ? one(ComplexF64) : zero(ComplexF64)
        for t in f.terms
            p = complex(t.μ)
            for b in t.blocks
                p *= A[b]
            end
            val += p
        end
        out[i] = isempty(f.ks) ? Ψ[f.centre] * val : w * Ψ[f.centre] * val
    end
    return out
end

"""
    evaluate(basis, P::Purification, env)

All basis functions at `env` via `P`.
"""
function evaluate(basis::dACEBasis, P::Purification, env::Environment)
    work = EvaluationWorkspace(basis, P)
    return evaluate!(work.values, basis, env, work, P)
end
