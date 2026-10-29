# fit, fit_path, design_statistics and select_lambda for the 2D bases (and composites), and the direct solver

"""
    design_matrix(basis, envs; works = nothing)

`B[s, k] = Bₖ(envs[s])`; `works`: workspaces to reuse (`workspace_pool(basis)`).
"""
function design_matrix(basis, envs::AbstractVector; works = nothing)
    B = zeros(ComplexF64, length(envs), length(basis))
    foreach_workspace(basis, length(envs); works) do work, s
        evaluate!(view(B, s, :), basis, envs[s], work)
    end
    return B
end

# a = Σ (αₖ + iβₖ) Bₖ: rows (Re; Im), columns (α; β)
function real_system(B::AbstractMatrix{<:Complex}, accs)
    A = [real(B) -imag(B); imag(B) real(B)]
    y = [first.(accs); last.(accs)]
    return A, y
end

"""
    fit(basis, envs, accs; λ = :evidence, λself = λ, prior = smoothness_prior(basis), reflections = false,
        self_fit = :staged, method = :qr, chunksize = 2000)
    fit(basis, S::QRStatistics; λ, λself, prior, reflections, self_fit)
    fit(basis, envs, accs; solver)

`min Σₛ |aₛ - Σₖ cₖ Bₖ(envₛ)|² + λ² Σₖ γₖ |c̃ₖ|²` (`c̃`: column-normalised basis, `γ = prior`: by default the
smoothness prior, `prior = nothing` for equal weights). `λ = :evidence` (default): the regularisation that
maximises the evidence (marginal likelihood) of the fit, as Bayesian ridge regression in ACE, stage by stage for a
staged fit, without validation data (`method = :qr`); a number fixes it. The model records the `λ` and `λself` it
was fitted with (`m.λ`). `reflections = true`: `c ∈ ℝᴷ`, i.e. the model is also
equivariant under reflections (most collective-motion models are; the default `false` allows chiral rules).
`method = :qr` (default): streamed QR of the design, one chunk of environments at a time, so that memory does
not grow with the data and the condition number is that of the design; `method = :normal_equations`: the
normal equations accumulated over chunks, which square the condition number but are cheaper to accumulate.
From streamed statistics `S` (`design_statistics`, pooled with `+`): the same fit without the data, e.g. after
`select_lambda`. With an ACEfit `solver`: a direct solve of the full design matrix (joint only, `reflections` honoured,
the regularisation that of the solver: `λ` and `λself` must be omitted, `prior` is not used). A `CompositeBasis`
is fitted by the streamed QR only (no `method`, no `solver`).

`self_fit` (for a basis with a `History`): `:staged` (default) fits the interaction functions first, alone,
with `λ`, then the self term on their residual with `λself`, then the reactions to neighbours that see the
agent's own past (`interactions = (own = ...,)`) on what is left, with `λ`. A self term with velocity history
can absorb interaction signal (the two are not identified), so the staged fit keeps the interactions of the
fit without any past. `:self_first`: the self term first, then the interactions, then the `own` terms (the
self term absorbs whatever interaction signal it can). `:joint`: all columns in one least-squares problem, as
ACE fits a potential.
"""
function fit(
        basis::dACEBasis, envs, accs;
        solver = nothing, λ = :evidence, λself = λ, prior = smoothness_prior(basis), chunksize = 2000, reflections = false,
        method = :qr, self_fit = :staged,
    )
    check_lengths(envs, accs)
    check_method(method)
    self_fit = resolve_self_fit(basis, self_fit)
    if !isnothing(solver)
        self_fit === :joint || throw(ArgumentError("an ACEfit solver fits jointly: pass self_fit = :joint"))
        λ === :evidence && λself === :evidence || throw(ArgumentError("an ACEfit solver sets its own regularisation: omit λ and λself"))
        return fit_direct(basis, envs, accs, solver; reflections)
    end
    targets = [complex(x[1], x[2]) for x in accs]
    c, reg = solve_targets(basis, envs, targets; λ, λself, prior, chunksize, reflections, method, self_fit)
    return dACEModel(basis, c, reg.λ, reg.λself)
end

function fit(basis, S::QRStatistics; λ = :evidence, λself = λ, prior = smoothness_prior(basis), reflections = false, self_fit = :staged)
    c, reg = solve_statistics(S, basis; λ, λself, prior, reflections, self_fit = resolve_self_fit(basis, self_fit))
    return dACEModel(basis, c, reg.λ, reg.λself)
end

# coefficients and the regularisation (; λ, λself) they were fitted with, `:evidence` resolved
function solve_statistics(S, basis; λ, λself, prior, reflections, self_fit)
    # by the evidence, the coefficients come from the same SVDs (no second solve, which would hold another
    # dense K × K array while the SVD's are alive)
    if λ === :evidence || (self_fit !== :joint && λself === :evidence)
        ev = evidence_lambdas(S, basis; λ, λself, prior, reflections, self_fit)
        return ev.c, (; ev.λ, ev.λself)
    end
    c = self_fit === :joint ? lsq_solve(S; λ, prior, reflections) : staged_solve(S, basis; λ, λself, prior, reflections, self_fit)
    return c, (; λ, λself = self_fit === :joint ? λ : λself)
end

"""
    design_statistics(basis, envs, accs; chunksize = 2000) -> QRStatistics

Streamed QR statistics of the design of `basis` and the accelerations `accs` (one data set, e.g. one video or
trial). Statistics of separate data sets pool with `+`, so a cross-validation needs one pass over each data
set: `fit(basis, S_train + S_validation; λ...)` after `select_lambda(S_train, S_validation, basis; λs)`.
For the 3D bases `(plane, axis)`: a pair of statistics (`FrameStatistics`).
"""
design_statistics(basis, envs, accs; chunksize = 2000) =
    qr_statistics(basis, envs, [complex(a[1], a[2]) for a in accs]; chunksize)

"""
    select_lambda(S, Sval, basis; λs, prior = smoothness_prior(basis), reflections = false, self_fit = :staged)
        -> (; λ, λself)

The regularisation that minimises the error on validation statistics `Sval` of the fit on `S`, for any
`self_fit`: joint, one `λ` over the path (`λself = λ`); staged, stage by stage (`select_staged`).
"""
function select_lambda(S::QRStatistics, Sval::QRStatistics, basis; λs, prior = smoothness_prior(basis), reflections = false, self_fit = :staged)
    self_fit = resolve_self_fit(basis, self_fit)
    if self_fit === :joint
        cs = lsq_path(S; λs, prior, reflections)
        λ = λs[argmin([lsq_sse(Sval, c) for c in cs])]
        return (; λ, λself = λ)
    end
    λ, λself = select_staged(S, Sval, basis; λs, prior, reflections, self_fit)
    return (; λ, λself)
end

function solve_targets(basis, envs, targets; λ, λself, prior, chunksize, reflections, method, self_fit)
    method === :qr && return solve_statistics(qr_statistics(basis, envs, targets; chunksize), basis; λ, λself, prior, reflections, self_fit)
    check_evidence_method(method, λ, λself)
    G, h = accumulate_normal_equations(basis, envs, targets; chunksize)
    c = self_fit === :joint ? solve_normal_equations(G, h; λ, prior, reflections) :
        staged_normal_equations(G, h, basis; λ, λself, prior, reflections, self_fit)
    return c, (; λ, λself = self_fit === :joint ? λ : λself)
end

"""
    fit_path(basis, envs, accs; λs, λself = nothing, prior = smoothness_prior(basis), reflections = false,
             self_fit = :staged, method = :qr, chunksize = 2000)

`fit` for each `λ ∈ λs` from one pass over the data: one streamed QR, then one SVD of its triangular
factor for a joint fit, or one SVD of the first stage of a staged fit and a ridge solve of the later stages per
`λ` (`method = :qr`), or one accumulation of the normal equations. Staged fits use `λself` for the self term
(default: the same `λ`), and `select_lambda` picks both on validation data.
"""
function fit_path(basis::dACEBasis, envs, accs; λs, prior = smoothness_prior(basis), chunksize = 2000, reflections = false, method = :qr,
        self_fit = :staged, λself = nothing)
    check_method(method)
    self_fit = resolve_self_fit(basis, self_fit)
    targets = [complex(x[1], x[2]) for x in accs]
    if method === :qr
        S = qr_statistics(basis, envs, targets; chunksize)
        cs = self_fit === :joint ? lsq_path(S; λs, prior, reflections) : staged_path(S, basis; λs, λself, prior, reflections, self_fit)
        return [dACEModel(basis, c, λ, effective_λself(self_fit, λself, λ)) for (c, λ) in zip(cs, λs)]
    end
    G, h = accumulate_normal_equations(basis, envs, targets; chunksize)
    solve(λ) = self_fit === :joint ? solve_normal_equations(G, h; λ, prior, reflections) :
        staged_normal_equations(G, h, basis; λ, λself = something(λself, λ), prior, reflections, self_fit)
    return [dACEModel(basis, solve(λ), λ, effective_λself(self_fit, λself, λ)) for λ in λs]
end

function fit_direct(basis, envs, accs, solver; reflections = false)
    bytes = 112 * length(envs) * length(basis)
    bytes > Sys.total_memory() / 4 && error(
        "direct solve needs ≈ $(round(bytes / 2^30; digits = 1)) GB; use the default " *
            "streamed QR fit (`method = :qr`) or fewer environments",
    )
    B = design_matrix(basis, envs)
    # reflections: real coefficients, a = Σ αₖ Bₖ, rows (Re; Im)
    A, y = reflections ? ([real(B); imag(B)], [first.(accs); last.(accs)]) : real_system(B, accs)
    s = map(norm, eachcol(A))
    s[s .== 0] .= 1
    A ./= s'
    θ = ACEfit.solve(solver, A, y)["C"] ./ s
    K = length(basis)
    return dACEModel(basis, reflections ? complex.(θ) : θ[1:K] + im * θ[(K + 1):end])
end

# as `fit` for a `dACEBasis`, by streamed QR only
function fit(basis::CompositeBasis, envs, accs; λ = :evidence, λself = λ, prior = smoothness_prior(basis), chunksize = 2000, reflections = false,
        self_fit = :staged)
    check_lengths(envs, accs)
    targets = [complex(x[1], x[2]) for x in accs]
    c, reg = solve_targets(basis, envs, targets; λ, λself, prior, chunksize, reflections, method = :qr,
        self_fit = resolve_self_fit(basis, self_fit))
    return dACEModel(basis, c, reg.λ, reg.λself)
end
