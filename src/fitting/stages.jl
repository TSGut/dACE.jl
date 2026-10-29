# staged fit: blocks of design columns one after another, each on the residual of those before it. Every column
# subset of the factor F has the Gram matrix of the same columns of the design, so all stages come from the
# statistics of the full design. A stage is (columns, tag): tag :interactions uses λ, :selfterm uses λself.
stage_prior(prior, cols) = isnothing(prior) ? nothing : prior[cols]

stage_lambda(tag, λ, λself) = tag === :selfterm ? λself : λ

function stage_update!(r, F, cols, c)
    x = eltype(F) <: Real ? real(c) : c
    r .-= view(F, :, cols) * x
    return r
end

stage_solve(F, cols, r; λ, prior) = ridge_solve(view(F, :, cols), r; λ, prior = stage_prior(prior, cols))

# the stages from the start (c = 0, r = targets) or continuing a partial solution (c, r)
function run_stages!(c, r, F, stages; λ, λself, prior)
    for (cols, tag) in stages
        isempty(cols) && continue
        cb = stage_solve(F, cols, r; λ = stage_lambda(tag, λ, λself), prior)
        c[cols] .= cb
        stage_update!(r, F, cols, cb)
    end
    return c
end

stage_start(F, K) = (zeros(ComplexF64, K), Vector{eltype(F)}(F[:, K + 1]))

"""
    staged_solve(S::QRStatistics, basis; λ = 1e-6, λself = λ, prior = nothing, reflections = false, self_fit = :staged) -> c

Coefficients of `fit(basis, ...; self_fit)` from the streamed statistics `S` of the full design of `basis`.
`:staged`: the interactions alone (`λ`), then the self term on their residual (`λself`), then, with
`History(interactions = (own = ...,))`, the reactions to neighbours that see the agent's own past (`λ`); the
interaction coefficients of the first stage are those of the fit without any past. `:self_first`: the self
term first, then the interactions, then the `own` terms.
"""
function staged_solve(S::QRStatistics, basis; λ = 1.0e-6, λself = λ, prior = nothing, reflections = false, self_fit = :staged)
    F, K = factor(check_plain(S), reflections), S.K
    K == length(basis) || throw(DimensionMismatch("statistics of $K columns for a basis of $(length(basis))"))
    c, r = stage_start(F, K)
    return run_stages!(c, r, F, fit_stages(basis, self_fit); λ, λself, prior)
end

"""
    staged_path(S::QRStatistics, basis; λs, λself = nothing, prior = nothing, reflections = false, self_fit = :staged)
        -> [c for λ in λs]

`staged_solve` for every interaction `λ ∈ λs`, the self term with `λself` (default: the same `λ`).
"""
function staged_path(S::QRStatistics, basis; λs, λself = nothing, prior = nothing, reflections = false, self_fit = :staged)
    F, K = factor(check_plain(S), reflections), S.K
    stages = filter(st -> !isempty(st[1]), fit_stages(basis, self_fit))
    (cols, tag), rest = stages[1], stages[2:end]
    if tag === :interactions   # one SVD for the first stage over all λ
        return map(zip(stage_path(F, cols, F[:, K + 1]; λs, prior), λs)) do (cb, λ)
            c, r = stage_start(F, K)
            c[cols] .= cb
            run_stages!(c, stage_update!(r, F, cols, cb), F, rest; λ, λself = something(λself, λ), prior)
        end
    end
    return [run_stages!(stage_start(F, K)..., F, stages; λ, λself = something(λself, λ), prior) for λ in λs]
end

stage_path(F, cols, r; λs, prior) = ridge_path(view(F, :, cols), r; λs, prior = stage_prior(prior, cols))

"""
    select_staged(S, Sval, basis; λs, prior = nothing, reflections = false, self_fit = :staged) -> (λ, λself)

Regularisation of `staged_solve` on validation statistics `Sval`, stage by stage: each stage's `λ` minimises the
validation error of the model up to that stage, so the first stage's `λ` is that of a fit with that stage only.
The `own` stage reuses the interactions' `λ`.
"""
function select_staged(S::QRStatistics, Sval::QRStatistics, basis; λs, prior = nothing, reflections = false, self_fit = :staged)
    F, K = factor(check_plain(S), reflections), S.K
    c, r = stage_start(F, K)
    chosen = Dict{Symbol, Float64}()
    for (cols, tag) in fit_stages(basis, self_fit)
        isempty(cols) && continue
        if haskey(chosen, tag)
            run_stages!(c, r, F, [(cols, tag)]; λ = chosen[tag], λself = chosen[tag], prior)
            continue
        end
        cbs = stage_path(F, cols, r; λs, prior)
        k = argmin([(c[cols] .= cb; lsq_sse(Sval, c)) for cb in cbs])
        chosen[tag] = λs[k]
        c[cols] .= cbs[k]
        stage_update!(r, F, cols, cbs[k])
    end
    λ = get(chosen, :interactions, get(chosen, :selfterm, first(λs)))
    return λ, get(chosen, :selfterm, λ)
end

"""
    column_blocks(basis) -> (interactions, own, selfterm)

Column indices of the interaction functions without past, those that see the agent's own past
(`History(interactions = (own = ...,))`), and the self term.
"""
function column_blocks(basis)
    hasproperty(basis, :funcs) || return (collect(1:length(basis)), Int[], Int[])
    K = length(basis.funcs)
    own = [sees_own_past(basis, f) for f in basis.funcs]
    return findall(!, own), findall(own), collect((K + 1):length(basis))
end

# the stages of a staged fit, in order
function fit_stages(basis, self_fit)
    check_self_fit(self_fit)
    self_fit === :joint && throw(ArgumentError("self_fit = :joint has no stages: use lsq_solve"))
    plain, own, selfterm = column_blocks(basis)
    first2 = self_fit === :staged ? [(plain, :interactions), (selfterm, :selfterm)] : [(selfterm, :selfterm), (plain, :interactions)]
    return [first2; (own, :interactions)]
end

# a basis with fewer than two non-empty blocks fits the same either way: :joint
resolve_self_fit(basis, self_fit) = (check_self_fit(self_fit); count(!isempty, column_blocks(basis)) >= 2 ? self_fit : :joint)

function column_blocks(b::CompositeBasis)
    plain, own, selfterm = column_blocks(b.interaction)
    return plain, own, [selfterm; (length(b.interaction) + 1):length(b)]
end

# the λself a fit actually used: a joint fit has one λ for all columns
effective_λself(self_fit, λself, λ) = self_fit === :joint ? λ : something(λself, λ)
