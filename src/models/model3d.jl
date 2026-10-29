# the 3D model: the plane and axis parts in the frame of the bases

"""
    FrameModel(plane, c_plane, axis, c_axis, λ = NaN, λself = λ)

`aᵢ = Re(a⊥) e₁ + Im(a⊥) e₂ + a∥ e₃` in the frame of the bases, `a⊥ = Σ c_plane B_plane`,
`a∥ = Re Σ c_axis B_axis`. `λ`, `λself`: the regularisation of the fit, one per part `(plane = ..., axis = ...)`.
"""
struct FrameModel{TA, TB}
    plane::TA
    c_plane::Vector{ComplexF64}
    axis::TB
    c_axis::Vector{ComplexF64}
    λ::NamedTuple{(:plane, :axis), Tuple{Float64, Float64}}
    λself::NamedTuple{(:plane, :axis), Tuple{Float64, Float64}}
    function FrameModel(plane::TA, c_plane, axis::TB, c_axis, λ = NaN, λself = λ) where {TA, TB}
        check_pair((plane, axis))
        length(c_plane) == length(plane) && length(c_axis) == length(axis) ||
            throw(DimensionMismatch("coefficients ($(length(c_plane)), $(length(c_axis))) for bases of ($(length(plane)), $(length(axis)))"))
        return new{TA, TB}(plane, c_plane, axis, c_axis, perpart(λ), perpart(λself))
    end
end

# the settings the two parts share: the frame and target projection, the neighbour search and the past states
function check_pair((plane, axis))
    plane.charge == 1 && axis.charge == 0 ||
        throw(ArgumentError("the plane basis needs charge 1 and the axis basis charge 0, got $(plane.charge) and $(axis.charge) (heading_bases gives the pair in order)"))
    plane.vertical == axis.vertical && plane.cutoff == axis.cutoff && plane.history == axis.history ||
        throw(ArgumentError("the plane and axis bases must share the frame (vertical), cutoff and history"))
    return nothing
end

# a prior for both parts: one per part, or `nothing` (equal weights) for both
frame_prior(prior) = isnothing(prior) ? (nothing, nothing) : prior

perpart(x::NamedTuple) = (plane = Float64(x.plane), axis = Float64(x.axis))

perpart(x) = (plane = Float64(x), axis = Float64(x))

# from the coefficients and regularisation of each part
FrameModel(plane, (cp, rp)::Tuple, axis, (ca, ra)::Tuple) =
    FrameModel(plane, cp, axis, ca, (plane = rp.λ, axis = ra.λ), (plane = rp.λself, axis = ra.λself))

cutoff(m::FrameModel) = m.plane.cutoff

has_history(m::FrameModel) = !isnothing(m.plane.history)

history_of(m::FrameModel) = m.plane.history

function predict(m::FrameModel, env::Environment3)
    e1, e2, e3 = basis_frame(m.plane, env)
    a_plane = contract(evaluate(m.plane, env), m.c_plane)
    a_ax = real(contract(evaluate(m.axis, env), m.c_axis))
    return real(a_plane) * e1 + imag(a_plane) * e2 + a_ax * e3
end

function body_order_contributions(m::FrameModel, env::Environment3)
    e1, e2, e3 = basis_frame(m.plane, env)
    out = fill(zero(SVector{3, Float64}), max(max_order(m.plane), max_order(m.axis)) + 1)
    for (k, (o, B)) in enumerate(zip(correlation_orders(m.plane), evaluate(m.plane, env)))
        z = m.c_plane[k] * B
        out[o + 1] += real(z) * e1 + imag(z) * e2
    end
    for (k, (o, B)) in enumerate(zip(correlation_orders(m.axis), evaluate(m.axis, env)))
        out[o + 1] += real(m.c_axis[k] * B) * e3
    end
    return out
end
