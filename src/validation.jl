# checks of user input shared by the whole package, at the public boundaries: environments, basis options, data,
# fit options and regularisation; each throws an ArgumentError or DimensionMismatch that names the input

# ---------------------------------------------------------------- environments (2D and 3D)

# r, u and (if given) past_vn: one entry per neighbour; no neighbour at r = 0, whose direction is undefined
function check_environment(r, u, past_vn)
    length(u) == length(r) && (isempty(past_vn) || length(past_vn) == length(r)) ||
        throw(DimensionMismatch("r, u and (if given) past_vn must have one entry per neighbour"))
    j = findfirst(iszero, r)
    j === nothing || throw(ArgumentError("neighbour $j is at distance 0 (coincident positions), its direction is undefined"))
    return nothing
end

# positions and velocities of a snapshot (dimension × agents) or a trajectory (dimension × agents × frames): one shape,
# in 2D or 3D
function check_snapshot(pos, vel)
    size(pos) == size(vel) || throw(DimensionMismatch("positions of size $(size(pos)) but velocities of size $(size(vel))"))
    size(pos, 1) in (2, 3) || throw(ArgumentError("only 2D and 3D data are supported, got dimension $(size(pos, 1))"))
    return nothing
end

positive_dt(dt) = (dt isa Real && isfinite(dt) && dt > 0) || throw(ArgumentError("the sampling step dt must be positive and finite, got $dt"))

# ---------------------------------------------------------------- bases

# the lengths and scales a basis divides by: positive and finite
function check_scales(; kwargs...)
    for (name, x) in kwargs
        (x isa Real && isfinite(x) && x > 0) || throw(ArgumentError("$name must be positive and finite, got $x"))
    end
    return nothing
end

# the options every basis constructor shares (dACEBasis, FrameBasis)
function check_basis_options(; inputmap, history, scales...)
    inputmap in INPUTMAPS || throw(ArgumentError("inputmap must be one of $INPUTMAPS"))
    isnothing(history) || history isa History || throw(ArgumentError("history must be a History or nothing (see History)"))
    check_scales(; scales...)
    return nothing
end

# ---------------------------------------------------------------- data

check_lengths(envs, accs) = length(envs) == length(accs) ||
    throw(DimensionMismatch("$(length(envs)) environments but $(length(accs)) targets"))

# all of several data vectors (e.g. the current and lagged data of the AR(1) fit) of one length
check_lengths(xs...) = allequal(length.(xs)) ||
    throw(DimensionMismatch("data of different lengths: $(join(length.(xs), ", "))"))

# ---------------------------------------------------------------- fit options and regularisation

const FIT_METHODS = (:qr, :normal_equations)

check_method(method) = method in FIT_METHODS ||
    throw(ArgumentError("method must be :qr (streamed QR, default) or :normal_equations"))

const SELF_FITS = (:staged, :self_first, :joint)

check_self_fit(x) = x in SELF_FITS || throw(ArgumentError(
    "self_fit must be :staged (interactions first, then the self term, default), :self_first (the reverse) or :joint"))

# the evidence needs the SVD of the streamed factor, which the normal equations do not provide
check_evidence_method(method, λs...) = method === :normal_equations && any(λ -> λ === :evidence, λs) &&
    throw(ArgumentError("λ = :evidence needs method = :qr: pass a number for λ with the normal equations"))

# ridge regularisation is strictly positive (λ = 0 has no unique solution for a rank-deficient design), and so are
# the prior weights (a zero weight leaves a column unpenalised, which the scaled path cannot represent)
check_lambda(λ) = (λ isa Real && isfinite(λ) && λ > 0) ||
    throw(ArgumentError("λ must be a positive finite number (or :evidence in fit), got $λ"))

function prior_weights(prior, K)
    γ = isnothing(prior) ? ones(K) : prior
    length(γ) == K || throw(DimensionMismatch("prior has length $(length(γ)), expected $K"))
    all(g -> isfinite(g) && g > 0, γ) || throw(ArgumentError("prior weights must be positive and finite"))
    return γ
end
