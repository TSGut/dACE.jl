# the AR(1) noise diagnostic: ARModel, Cochrane–Orcutt fits on QR statistics and on normal equations

"""
    ARModel(model, ρ)

`aᵢ(t) = F(envᵢ(t)) + ηᵢ(t)`, `ηᵢ(t) = ρ ηᵢ(t - Δ) + ξ`: force law `F = model` and
persistent individual noise (from `fit_ar1`). `predict(m, env)`: the force law;
`predict(m, env, lagged_env, lagged_acc)`: one step ahead given the environment and acceleration a step `Δ` earlier.
"""
struct ARModel{M}
    model::M
    ρ::Float64
end

predict(m::ARModel, env) = predict(m.model, env)

predict(m::ARModel, envs::AbstractVector) = predict(m.model, envs)

predict(m::ARModel, env, penv, pacc) = predict(m.model, env) + m.ρ * (pacc - predict(m.model, penv))

cutoff(m::ARModel) = cutoff(m.model)

history_of(m::ARModel) = history_of(m.model)

"""
    ar_statistics(basis, envs, penvs, accs, paccs; chunksize = 2000, method = :qr)

Statistics of the AR(1) noise model for `ar_solve` and `ar_sse`: the streamed QR of the current
and lagged designs and targets (`qr_statistics` with `past`), or with
`method = :normal_equations` their cross products (`ar_cross_products`).
"""
function ar_statistics(basis, envs, penvs, accs, paccs; chunksize = 2000, method = :qr)
    check_method(method)
    method === :normal_equations && return ar_cross_products(basis, envs, penvs, accs, paccs; chunksize)
    cx = a -> complex(a[1], a[2])
    return qr_statistics(basis, envs, cx.(accs); past = (penvs, cx.(paccs)), chunksize)
end

# AR(1) on the lagged factor: quasi-differenced design and targets for a given ρ
function ar_blocks(F, K, ρ)
    A = view(F, :, 1:K) .- ρ .* view(F, :, (K + 1):(2K))
    b = view(F, :, 2K + 1) .- ρ .* view(F, :, 2K + 2)
    return A, b
end

# Cochrane–Orcutt: alternate the quasi-differenced ridge fit and ρ from the residuals. `λ = :evidence`: each fit
# chooses its λ by the evidence of the quasi-differenced system (the noise ξ of the AR(1) model is independent).
ar_solve(S::QRStatistics; kwargs...) = (r = ar_solve_λ(S; kwargs...); (r[1], r[2]))

function ar_solve_λ(S::QRStatistics; λ = :evidence, prior = nothing, reflections = false, iters = 50, tol = 1.0e-8)
    S.lagged || throw(ArgumentError("ar_solve needs lagged statistics (qr_statistics with past)"))
    F, K = factor(S, reflections), S.K
    B, Bp, t, tp = view(F, :, 1:K), view(F, :, (K + 1):(2K)), view(F, :, 2K + 1), view(F, :, 2K + 2)
    n = reflections ? 2S.n : S.n
    solve(ρ) = λ === :evidence ? evidence_solve(ar_blocks(F, K, ρ)..., n; prior) : (λ, ridge_solve(ar_blocks(F, K, ρ)...; λ, prior))
    ρ = 0.0
    λρ, c = solve(ρ)
    for _ in 1:iters
        e, ep = t - B * c, tp - Bp * c
        ρnew = persistence(real(dot(ep, e)), real(dot(ep, ep)), real(dot(tp, tp)))
        λρ, c = solve(ρnew)
        abs(ρnew - ρ) < tol && (ρ = ρnew; break)
        ρ = ρnew
    end
    return c, ρ, λρ
end

# the persistence ρ = ⟨e(t - Δ), e(t)⟩ / |e(t - Δ)|², clamped to (-1, 1); when the lagged residual vanishes
# (relative to the lagged targets, `scale`) ρ is not identified and taken as 0
function persistence(num, den, scale)
    den > 1.0e-12 * max(scale, floatmin()) || return 0.0
    return clamp(num / den, -0.99, 0.99)
end

ar_sse(S::QRStatistics, c, ρ) =
    (S.lagged || throw(ArgumentError("ar_sse needs lagged statistics")); sum(abs2, S.R * [-c; ρ .* c; 1; -ρ]))

"""
    ar_cross_products(basis, envs, penvs, accs, paccs; chunksize = 2000)

Cross products of the current (`B`, `a`) and lagged (`Bₚ`, `aₚ`) designs and targets, from
which the quasi-differenced normal equations follow for any `ρ`
(`ar_statistics(...; method = :normal_equations)`).
"""
function ar_cross_products(basis, envs, penvs, accs, paccs; chunksize = 2000)
    K = length(basis)
    G, Gp, X = (zeros(ComplexF64, K, K) for _ in 1:3)
    h, hp, ph, pp = (zeros(ComplexF64, K) for _ in 1:4)
    yy = zero(ComplexF64); pyy = zero(ComplexF64); py = zero(ComplexF64)
    t = [complex(a[1], a[2]) for a in accs]
    tp = [complex(a[1], a[2]) for a in paccs]
    works = workspace_pool(basis)
    for chunk in Iterators.partition(eachindex(envs), chunksize)
        B, Bp = design_matrix(basis, view(envs, chunk); works), design_matrix(basis, view(penvs, chunk); works)
        y, yp = view(t, chunk), view(tp, chunk)
        mul!(G, B', B, true, true); mul!(Gp, Bp', Bp, true, true); mul!(X, Bp', B, true, true)
        mul!(h, B', y, true, true); mul!(hp, B', yp, true, true); mul!(ph, Bp', y, true, true); mul!(pp, Bp', yp, true, true)
        yy += y' * y; pyy += yp' * yp; py += yp' * y
    end
    return (; G, Gp, X, h, hp, ph, pp, yy = real(yy), pyy = real(pyy), py)
end

ar_normal_equations(S, ρ) = (S.G - ρ * (S.X + S.X') + ρ^2 * S.Gp, S.h - ρ * (S.hp + S.ph) + ρ^2 * S.pp)

# Cochrane–Orcutt: alternate the quasi-differenced fit and ρ from the residuals
function ar_solve(S; λ = 1.0e-6, prior = nothing, reflections = false, iters = 50, tol = 1.0e-8)
    ρ = 0.0
    c = solve_normal_equations(ar_normal_equations(S, ρ)...; λ, prior, reflections)
    for _ in 1:iters
        num = real(S.py - S.hp' * c - c' * S.ph + c' * S.X * c)
        den = real(S.pyy - 2real(c' * S.pp) + c' * S.Gp * c)
        ρnew = persistence(num, den, real(S.pyy))
        c = solve_normal_equations(ar_normal_equations(S, ρnew)...; λ, prior, reflections)
        abs(ρnew - ρ) < tol && (ρ = ρnew; break)
        ρ = ρnew
    end
    return c, ρ
end

# one-step-ahead squared error of (c, ρ) on statistics S
ar_sse(S, c, ρ) = (ar = ar_normal_equations(S, ρ); real(S.yy - 2ρ * real(S.py) + ρ^2 * S.pyy - 2real(c' * ar[2]) + c' * ar[1] * c))

"""
    fit_ar1(basis, envs, accs, lagged_envs, lagged_accs; λ = :evidence, prior = smoothness_prior(basis),
            reflections = false, method = :qr, chunksize = 2000) -> ARModel

Diagnostic, not for interaction laws: the force law fitted jointly with persistent individual noise
`η(t) = ρ η(t - Δ) + ξ` by Cochrane–Orcutt iteration (`lagged_envs`, `lagged_accs`: the same agents one step
`Δ` earlier, as from `trajectory_data(...; lagged = true)`); all columns jointly. The fitted `ρ` shows whether
residuals persist. With tracked positions the previous finite-difference acceleration shares frames with the
target, so the one-step prediction of an `ARModel` partly predicts tracking error, and as `ρ → 1` the force law
is not identified: judge the force law (`predict(m.model, env)`), not the one-step prediction. `λ = :evidence`
(default, `method = :qr`): each Cochrane–Orcutt fit chooses λ by the evidence of its quasi-differenced system;
`m.model.λ` is the last one.
"""
function fit_ar1(basis::dACEBasis, envs, accs, lagged_envs, lagged_accs; λ = :evidence, prior = smoothness_prior(basis),
        reflections = false, method = :qr, chunksize = 2000)
    check_lengths(envs, accs, lagged_envs, lagged_accs)
    check_method(method)
    check_evidence_method(method, λ)
    S = ar_statistics(basis, envs, lagged_envs, accs, lagged_accs; chunksize, method)
    c, ρ, λ = method === :qr ? ar_solve_λ(S; λ, prior, reflections) : (ar_solve(S; λ, prior, reflections)..., λ)
    return ARModel(dACEModel(basis, c, λ), ρ)
end
