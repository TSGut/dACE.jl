# purification of the 3D basis (as purification2d.jl, in the frame of the basis)

"""
    Purification3

`A = L * Ã`, `Ã = Σⱼ fc^p P_ν(y) P_α(t) P_λ(z) P_β(ũ₃) w^M ũ⊥^μ |b̃⊥|^{2Q} b̃⊥^K b̃₃^J`;
`|w|² = 1 - t²`, `|ũ⊥|² = (z + 1)/2`.
"""
struct Purification3{TL}
    pooled::Vector{NTuple{10, Int}}
    L::SparseMatrixCSC{Float64, Int}
    poly::TL
    Mmax::Int
    μmax::Int
    pmax::Int
    Qmax::Int
    Kmax::Int
    Jmax::Int
end

function purification(basis::FrameBasis)
    onep = basis.onep
    q(β, f) = (sum(abs(f(onep[k])) for k in β) - abs(sum(f(onep[k]) for k in β))) ÷ 2
    degs(β) = (
        sum(onep[k].n for k in β), sum(onep[k].a for k in β) + 2q(β, φ -> φ.mr),
        sum(onep[k].l for k in β) + q(β, φ -> φ.mu), sum(onep[k].b for k in β),
    )
    dmax = maximum(β -> maximum(degs(β)), basis.blocks; init = 1)
    Lext = legendre_basis(max(dmax, 1) + 1)
    n = dmax + 8
    x = [cos(π * (i - 0.5) / n) for i in 1:n]
    V = P4ML.evaluate(basis.poly, x)
    F = qr(P4ML.evaluate(Lext, x), ColumnNorm())
    prodcol(β, f) = reduce(.*, (V[:, f(onep[k]) + 1] for k in β))

    index = Dict{NTuple{10, Int}, Int}()
    pooled = NTuple{10, Int}[]
    I, J, W = Int[], Int[], Float64[]
    for (bidx, β) in enumerate(basis.blocks)
        p = length(β)
        M = sum(onep[k].mr for k in β)
        μ = sum(onep[k].mu for k in β)
        # ∏ₜ b̃⊥^{kₜ} b̃₃^{jₜ} = |b̃⊥|^{2Q} b̃⊥^K b̃₃^J
        nb = (q(β, φ -> φ.k), sum(onep[k].k for k in β), sum(onep[k].j for k in β))
        cr = legendre_coefficients(F, prodcol(β, φ -> φ.n))
        ct = legendre_coefficients(F, prodcol(β, φ -> φ.a) .* (1 .- x .^ 2) .^ q(β, φ -> φ.mr))
        cz = legendre_coefficients(F, prodcol(β, φ -> φ.l) .* ((x .+ 1) ./ 2) .^ q(β, φ -> φ.mu))
        cb = legendre_coefficients(F, prodcol(β, φ -> φ.b))
        for ν in findall(!iszero, cr), α in findall(!iszero, ct), λ in findall(!iszero, cz), b in findall(!iszero, cb)
            key = (p, ν - 1, α - 1, λ - 1, b - 1, M, μ, nb...)
            f = get!(index, key) do
                push!(pooled, key)
                length(pooled)
            end
            push!(I, bidx); push!(J, f); push!(W, cr[ν] * ct[α] * cz[λ] * cb[b])
        end
    end
    L = sparse(I, J, W, length(basis.blocks), length(pooled))
    Mmax = maximum(k -> abs(k[6]), pooled; init = 0)
    μmax = maximum(k -> abs(k[7]), pooled; init = 0)
    pmax = maximum(length, basis.blocks; init = 1)
    Qmax, Kmax, Jmax = (maximum(k -> abs(k[i]), pooled; init = 0) for i in 8:10)
    return Purification3(pooled, L, Lext, Mmax, μmax, pmax, Qmax, Kmax, Jmax)
end

function pooled_features(basis::FrameBasis, P::Purification3, env::Environment3)
    return pooled_features!(EvaluationWorkspace(basis, P), basis, P, env)
end

function pooled_features!(work, basis::FrameBasis, P::Purification3, env::Environment3)
    Ã, fcp, R, U = work.pooled, work.fcp, work.R, work.U
    Bq, Bk, Bj = work.Bq, work.Bk, work.Bj
    Ly, Lt, Lz, Lb = work.polys
    fill!(Ã, 0)
    length(env) == 0 && return Ã
    frame = basis_frame(basis, env)
    for j in 1:length(env)
        ρ, w, t, ũp, ũ₃, b̃p, b̃₃ = frame_coordinates(basis, frame, env, j)
        fc = fcut(ρ, basis.cutoff)
        fc == 0 && continue
        for p in 1:P.pmax
            fcp[p] = fc^p
        end
        P4ML.evaluate!(Ly, P.poly, radial_transform(basis, ρ))
        P4ML.evaluate!(Lt, P.poly, t)
        P4ML.evaluate!(Lz, P.poly, 2abs2(ũp) - 1)
        P4ML.evaluate!(Lb, P.poly, ũ₃)
        for m in -P.Mmax:P.Mmax
            R[m + P.Mmax + 1] = cpow(w, m)
        end
        for m in -P.μmax:P.μmax
            U[m + P.μmax + 1] = cpow(ũp, m)
        end
        for q in 0:P.Qmax
            Bq[q + 1] = abs2(b̃p)^q
        end
        for k in -P.Kmax:P.Kmax
            Bk[k + P.Kmax + 1] = cpow(b̃p, k)
        end
        for i in 0:P.Jmax
            Bj[i + 1] = b̃₃^i
        end
        for (f, (p, ν, α, λ, b, M, μ, Q, K, J)) in enumerate(P.pooled)
            Ã[f] += fcp[p] * Ly[ν + 1] * Lt[α + 1] * Lz[λ + 1] * Lb[b + 1] * R[M + P.Mmax + 1] * U[μ + P.μmax + 1] *
                Bq[Q + 1] * Bk[K + P.Kmax + 1] * Bj[J + 1]
        end
    end
    return Ã
end
