# CompositeBasis: an interaction basis with an external self model

"""
    CompositeBasis(interaction, selfterm)

The columns of `interaction` followed by those of another self model `selfterm`, both evaluated on the same
environments, fitted and predicted as one linear model (`fit`, `design_statistics`, `predict`). `selfterm` must
implement `length`, `evaluate!(out, selfterm, env, work)`, `evaluation_workspaces(selfterm)` and
`smoothness_prior(selfterm; p)`, and `basis_history(selfterm)` if it needs past states (for `rollout`); dACE's own self term is part of `dACEBasis(; history)` and needs no composite.
"""
struct CompositeBasis{A, S}
    interaction::A
    selfterm::S
end

Base.length(b::CompositeBasis) = length(b.interaction) + length(b.selfterm)

function evaluate!(out, b::CompositeBasis, env, work::Tuple)
    length(out) == length(b) || throw(DimensionMismatch("output length must match basis"))
    K = length(b.interaction)
    evaluate!(view(out, 1:K), b.interaction, env, work[1])
    evaluate!(view(out, (K + 1):length(b)), b.selfterm, env, work[2])
    return out
end

evaluate(b::CompositeBasis, env) = evaluate!(zeros(ComplexF64, length(b)), b, env, evaluation_workspaces(b))

evaluation_workspaces(b::CompositeBasis) = (evaluation_workspaces(b.interaction), evaluation_workspaces(b.selfterm))

smoothness_prior(b::CompositeBasis; p = 4) = [smoothness_prior(b.interaction; p); smoothness_prior(b.selfterm; p)]

# the history a composite needs: that of the interaction basis or of the self model (`basis_history(selfterm)`,
# `nothing` unless the self model defines it); both only when they agree
function basis_history(b::CompositeBasis)
    hi, hs = basis_history(b.interaction), basis_history(b.selfterm)
    isnothing(hs) && return hi
    (isnothing(hi) || hi == hs) && return hs
    throw(ArgumentError("the interaction basis and the self model need different histories: give them the same History"))
end
