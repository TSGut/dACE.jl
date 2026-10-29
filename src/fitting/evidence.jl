# Regularisation by the evidence (type II maximum likelihood, as Bayesian ridge regression in ACE): targets
# b = A c + ε with noise of precision β and a Gaussian prior of precision α on the scaled coefficients x = s√γ c,
# λ² = α/β. With β at its optimum n/Q, the log evidence is, up to a constant, K log λ² - Σᵢ log(λ² + σᵢ²) - n log Q(λ)
# (σᵢ: singular values of the scaled design, Q(λ): the minimum of the ridge objective, n: the number of data, complex
# data for complex coefficients, real data for real ones). Maximised over a grid of 0.01 decades, O(K) per point.
const EVIDENCE_GRID = 10.0 .^ (-14:0.01:3)

# the profiled log evidence on the grid `λs`
function log_evidence(D, b, n; λs = EVIDENCE_GRID)
    σ2, β2 = D.S .^ 2, abs2.(D.β)
    out = sum(abs2, b - D.U * D.β)   # the part of the targets outside the range of the design (no cancellation)
    logev(λ) = length(σ2) * log(λ^2) - sum(log.(σ2 .+ λ^2)) - n * log(max(out + λ^2 * sum(β2 ./ (σ2 .+ λ^2)), floatmin()))
    return logev.(λs)
end

evidence_lambda(D, b, n; λs = EVIDENCE_GRID) = λs[argmax(log_evidence(D, b, n; λs))]

# one λ for several independent data sets of the same (joint) model, e.g. the halves of a consistency check:
# the maximum of the summed evidence
function shared_evidence_lambda(Ss; prior = nothing, reflections = false, λs = EVIDENCE_GRID)
    total = sum(Ss) do S
        F, K = factor(check_plain(S), reflections), S.K
        b = F[:, K + 1]
        log_evidence(scaled_svd(view(F, :, 1:K), b; prior), b, reflections ? 2S.n : S.n; λs)
    end
    return λs[argmax(total)]
end

function evidence_solve(A, b, n; prior = nothing)
    D = scaled_svd(A, b; prior)
    λ = evidence_lambda(D, b, n)
    return λ, svd_coefficients(D, λ)
end

"""
    evidence_staged(S, basis; prior = nothing, reflections = false, self_fit = :staged, fixed = Dict()) -> (λ, λself, c)

Regularisation of `staged_solve` by the evidence of each stage on the residual of the stages before it (the `own`
stage reuses the interactions' `λ`), without validation data. `fixed`: tag (`:interactions`, `:selfterm`) => a
given `λ`, used as is. `c`: the coefficients of the staged fit with them, from the same SVDs.
"""
function evidence_staged(S::QRStatistics, basis; prior = nothing, reflections = false, self_fit = :staged,
        fixed = Dict{Symbol, Float64}())
    F, K = factor(check_plain(S), reflections), S.K
    n = reflections ? 2S.n : S.n
    c, r = stage_start(F, K)
    chosen = Dict{Symbol, Float64}(fixed)
    for (cols, tag) in fit_stages(basis, self_fit)
        isempty(cols) && continue
        if haskey(chosen, tag)
            run_stages!(c, r, F, [(cols, tag)]; λ = chosen[tag], λself = chosen[tag], prior)
            continue
        end
        λ, cb = evidence_solve(view(F, :, cols), r, n; prior = stage_prior(prior, cols))
        chosen[tag] = λ
        c[cols] .= cb
        stage_update!(r, F, cols, cb)
    end
    λ = get(chosen, :interactions, get(chosen, :selfterm, 1.0e-6))
    return λ, get(chosen, :selfterm, λ), c
end

# (λ, c) of the joint fit by the evidence
function evidence_joint(S::QRStatistics; prior = nothing, reflections = false)
    F, K = factor(check_plain(S), reflections), S.K
    return evidence_solve(view(F, :, 1:K), view(F, :, K + 1), reflections ? 2S.n : S.n; prior)
end

# the regularisation that maximises the evidence of the fit on `S` (`fit` with `λ = :evidence`) and the coefficients
# with it: joint, one `λ`; staged, one per stage, those given as numbers kept (and the later stages chosen on the
# residual they leave)
function evidence_lambdas(S::QRStatistics, basis; λ = :evidence, λself = λ, prior = smoothness_prior(basis), reflections = false,
        self_fit = :staged)
    self_fit = resolve_self_fit(basis, self_fit)
    if self_fit === :joint
        λ === :evidence || return (; λ, λself = λ, c = lsq_solve(S; λ, prior, reflections))
        λ, c = evidence_joint(S; prior, reflections)
        return (; λ, λself = λ, c)
    end
    fixed = Dict{Symbol, Float64}()
    λ === :evidence || (fixed[:interactions] = λ)
    λself === :evidence || (fixed[:selfterm] = λself)
    λ, λself, c = evidence_staged(S, basis; prior, reflections, self_fit, fixed)
    return (; λ, λself, c)
end
