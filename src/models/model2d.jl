# the 2D model: coefficients on a basis, prediction and body-order contributions

"""
    dACEModel(basis, c, λ = NaN, λself = λ)

`aₓ + i a_y = Σₖ cₖ Bₖ(env)`, `c ∈ ℂᴷ`, for a `dACEBasis` (with or without `history`) or a `CompositeBasis`.
`λ`, `λself`: the regularisation of the fit (`NaN` when not from `fit`).
"""
struct dACEModel{TB}
    basis::TB
    c::Vector{ComplexF64}
    λ::Float64
    λself::Float64
end

dACEModel(basis, c::AbstractVector, λ::Real = NaN, λself::Real = λ) =
    dACEModel(basis, Vector{ComplexF64}(c), Float64(λ), Float64(λself))

tovec(z::Complex) = SVector(real(z), imag(z))

"""
    predict(model, env)
    predict(model, envs::AbstractVector)

The acceleration a fitted model (`dACEModel`, `FrameModel` or `ARModel`) predicts for one environment, as an
`SVector`, or for each of a vector of environments (with one reusable workspace for the batch). For an `ARModel`,
`predict(model, env, lagged_env, lagged_acc)` adds the persistent noise one step ahead (see `ARModel`).
"""
predict(m::dACEModel, env::Environment) = tovec(contract(evaluate(m.basis, env), m.c))

cutoff(m::dACEModel) = m.basis.cutoff

"""
    body_order_contributions(model, env)

Contributions of the 1-, 2-, 3-body … terms (`[n]`: the `n`-body term, the self term in the 1-body term);
they sum to `predict(model, env)`.
"""
function body_order_contributions(m::dACEModel, env::Environment)
    B = evaluate(m.basis, env)
    out = zeros(ComplexF64, max_order(m.basis) + 1)
    for (k, o) in enumerate(correlation_orders(m.basis))   # the self term belongs to the 1-body term
        out[o + 1] += m.c[k] * B[k]
    end
    return tovec.(out)
end

has_history(m) = false

has_history(m::dACEModel) = !isnothing(history_of(m))

history_of(m) = nothing

history_of(m::dACEModel) = basis_history(m.basis)
