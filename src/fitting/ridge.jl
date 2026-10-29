# ridge solves on the streamed factor: one λ (augmented QR) or a path of λ (one SVD)

# min ‖A c - b‖² + λ² Σₖ γₖ sₖ² |cₖ|², sₖ = ‖A[:, k]‖ (columns that vanish get cₖ = 0), via the QR of
# the stacked system [A/s; λ√γ] (no normal equations)
function ridge_solve(A, b; λ, prior = nothing)
    check_lambda(λ)
    K = size(A, 2)
    s = [norm(view(A, :, k)) for k in 1:K]
    keep = findall(>(0), s)
    γ = prior_weights(prior, K)
    T = promote_type(eltype(A), eltype(b), Float64)
    M = zeros(T, size(A, 1) + length(keep), length(keep))
    for (j, k) in enumerate(keep)
        M[1:size(A, 1), j] .= view(A, :, k) ./ s[k]
        M[size(A, 1) + j, j] = λ * sqrt(γ[k])
    end
    rhs = [Vector{T}(b); zeros(T, length(keep))]
    x = qr!(M) \ rhs
    c = zeros(ComplexF64, K)
    c[keep] .= x ./ s[keep]
    return c
end

# The SVD of the scaled design [A/(s√γ)] and the projected targets, from which the fit follows for any λ. The scaled
# design is built once and the SVD works in place on it. Divide and conquer (xGESDD) is fast but occasionally fails to
# converge on ill-conditioned matrices; QR iteration (xGESVD) then takes over, on a fresh copy.
function scaled_svd(A, b; prior = nothing)
    K = size(A, 2)
    s = [norm(view(A, :, k)) for k in 1:K]
    keep = findall(>(0), s)
    γ = prior_weights(prior, K)
    w = s[keep] .* sqrt.(γ[keep])
    scaled() = (M = A[:, keep]; M ./= transpose(w); M)
    F = try
        svd!(scaled())
    catch err
        err isa LinearAlgebra.LAPACKException || rethrow()
        svd!(scaled(); alg = LinearAlgebra.QRIteration())
    end
    return (; K, keep, w, U = F.U, S = F.S, Vt = F.Vt, β = F.U' * b)
end

function svd_coefficients(D, λ)
    c = zeros(ComplexF64, D.K)
    c[D.keep] .= (D.Vt' * (D.S ./ (D.S .^ 2 .+ λ^2) .* D.β)) ./ D.w
    return c
end

# the same for every λ ∈ λs from one SVD of [A/(s√γ)]
ridge_path(A, b; λs, prior = nothing) = (foreach(check_lambda, λs); D = scaled_svd(A, b; prior); [svd_coefficients(D, λ) for λ in λs])

"""
    lsq_solve(S::QRStatistics; λ = 1e-6, prior = nothing, reflections = false) -> c

Coefficients of `fit` from the streamed QR statistics `S`.
"""
function lsq_solve(S::QRStatistics; λ = 1.0e-6, prior = nothing, reflections = false)
    F, K = factor(check_plain(S), reflections), S.K
    return ridge_solve(view(F, :, 1:K), view(F, :, K + 1); λ, prior)
end

"""
    lsq_path(S::QRStatistics; λs, prior = nothing, reflections = false) -> [c for λ in λs]

Coefficients for every `λ` from one SVD of the triangular factor.
"""
function lsq_path(S::QRStatistics; λs, prior = nothing, reflections = false)
    F, K = factor(check_plain(S), reflections), S.K
    return ridge_path(view(F, :, 1:K), view(F, :, K + 1); λs, prior)
end
