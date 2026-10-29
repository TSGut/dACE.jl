# choosing among candidate bases on validation data (select_basis), with a shared design for degree ladders

# the fields of a struct as a tuple, a hashable key
astuple(x) = ntuple(i -> getfield(x, i), fieldcount(typeof(x)))

# Every setting other than the degrees that changes the values of a basis function: in two bases with equal settings,
# functions with equal keys (`function_key`) are the same function. A new basis setting that changes values must be
# added here; bases without settings (`nothing`) never share a design.
ladder_settings(b) = nothing
ladder_settings(b::dACEBasis) = (:dACEBasis, b.cutoff, b.rscale, b.vscale, b.ascale, b.inputmap, b.history,
    b.inner_envelope, b.charge, b.normaliser, isnothing(b.purification), self_settings(b.selfterm))
ladder_settings(b::FrameBasis) = (:FrameBasis, b.cutoff, b.rscale, b.vscale, b.ascale, b.charge, b.vertical, b.inputmap,
    b.history, b.normaliser, isnothing(b.purification), self_settings(b.selfterm))
self_settings(::Nothing) = nothing
self_settings(s::SelfTerm) = (s.dim, s.channels, s.scales, s.vscale, s.inputmap, [(f.n, f.m0, f.terms) for f in s.funcs])

function function_key(basis, f)
    blk(b) = sort([astuple(basis.onep[k]) for k in basis.blocks[b]])
    terms = sort([(t.μ, sort(blk.(t.blocks))) for t in f.terms])
    return (astuple(basis.centres[f.centre]), sort([astuple(basis.onep[k]) for k in f.ks]), terms)
end

# Columns of a smaller basis inside a larger one built with the same settings (a degree ladder): a basis function is
# its centre, its one-particle factors and its purified product terms, all exact polynomial algebra, so the same
# function has the same values whatever the maximum degree. `nothing` when `small` is not a column subset of `large`.
function column_map(small, large)
    settings = ladder_settings(small)
    (isnothing(settings) || settings != ladder_settings(large)) && return nothing
    pos = Dict(function_key(large, f) => i for (i, f) in enumerate(large.funcs))
    cols = [get(pos, function_key(small, f), 0) for f in small.funcs]
    any(iszero, cols) && return nothing
    nself(small) == nself(large) || return nothing
    return [cols; length(large.funcs) .+ (1:nself(large))]
end

function column_map(small::Tuple, large::Tuple)
    length(small) == length(large) || return nothing
    maps = map(column_map, small, large)
    return any(isnothing, maps) ? nothing : maps
end

restrict_statistics(S::QRStatistics, cols) = select_columns(S, cols)

# statistics (training, validation) of every candidate: when all candidates are column subsets of the largest,
# the design of the largest only, restricted to the columns of each
function candidate_statistics(bases, envs, accs, val_envs, val_accs)
    L = argmax(b -> sum(length, b isa Tuple ? b : (b,)), bases)
    maps = [column_map(b, L) for b in bases]
    if !any(isnothing, maps) && length(bases) > 1
        S, Sv = design_statistics(L, envs, accs), design_statistics(L, val_envs, val_accs)
        return [(restrict_statistics(S, m), restrict_statistics(Sv, m)) for m in maps]
    end
    return [(design_statistics(b, envs, accs), design_statistics(b, val_envs, val_accs)) for b in bases]
end

"""
    select_basis(bases, envs, accs, val_envs, val_accs; λs, prior = smoothness_prior, reflections = false,
                 self_fit = :staged) -> (; basis, λ, λself, index, errors, statistics)

The candidate basis (for instance a ladder of total degrees, `dACEBasis(maxdeg = d, ...)` for several `d`) with
the lowest relative error on the validation data, each candidate with its own regularisation chosen on the same
validation data by `select_lambda`. `prior` is a function of the basis (default `smoothness_prior`) or `nothing`.
`errors` are the validation errors of all candidates. When the candidates differ only in degree, the design is
built once, for the largest. `statistics` are those of the chosen basis on training and validation data together,
for the refit on all data: `fit(basis, statistics; λ, λself)`. Passing a single candidate fixes the basis.
"""
function select_basis(bases, envs, accs, val_envs, val_accs; λs, prior = smoothness_prior, reflections = false, self_fit = :staged)
    isempty(bases) && throw(ArgumentError("no candidate bases"))
    stats = candidate_statistics(bases, envs, accs, val_envs, val_accs)
    results = map(bases, stats) do b, (S, Sv)
        # the default prior is left to select_lambda and fit, which know it for every kind of basis (also 3D pairs)
        kw = prior === smoothness_prior ? (; reflections, self_fit) : (; prior = prior isa Function ? prior(b) : prior, reflections, self_fit)
        sel = select_lambda(S, Sv, b; λs, kw...)
        m = fit(b, S; sel..., kw...)
        (; err = rmse(m, val_envs, val_accs; relative = true), sel...)
    end
    k = argmin([r.err for r in results])
    return (; basis = bases[k], λ = results[k].λ, λself = results[k].λself, index = k, errors = [r.err for r in results],
        statistics = stats[k][1] + stats[k][2])
end
