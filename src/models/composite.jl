# models on a CompositeBasis

prediction_workspace(m::dACEModel{<:CompositeBasis}) = (evaluation_workspaces(m.basis), zeros(ComplexF64, length(m.basis)))

predict_with_workspace(m::dACEModel{<:CompositeBasis}, env, work) = tovec(contract(evaluate!(work[2], m.basis, env, work[1]), m.c))

cutoff(m::dACEModel{<:CompositeBasis}) = m.basis.interaction.cutoff

# the self model belongs to the 1-body term
function body_order_contributions(m::dACEModel{<:CompositeBasis}, env::Environment)
    K = length(m.basis.interaction)
    out = body_order_contributions(dACEModel(m.basis.interaction, m.c[1:K]), env)
    out[1] += tovec(contract(evaluate!(zeros(ComplexF64, length(m.basis.selfterm)), m.basis.selfterm, env,
        evaluation_workspaces(m.basis.selfterm)), m.c[(K + 1):end]))
    return out
end
