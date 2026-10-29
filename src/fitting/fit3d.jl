# fitting the 3D pair of bases: per-part statistics, λ and fits

"""
    fit((plane, axis), envs, accs; λ = :evidence, λself = λ, prior = smoothness_prior.((plane, axis)),
        reflections = false, self_fit = :staged, method = :qr, chunksize = 2000)
    fit((plane, axis), S::FrameStatistics; λ, λself, prior, reflections, self_fit)

`fit` of the parts of `accs` in the plane and along the axis of each agent's frame, as the 2D `fit`;
`prior`: one weight vector per part. `λ`, `λself`: `:evidence` (default, chosen per part), a number, or one
per part `(plane = ..., axis = ...)` (as returned by `select_lambda`). `reflections = true`: real coefficients, the model also equivariant under
reflections through planes containing the axis.
"""
function fit(bases::Tuple{FrameBasis, FrameBasis}, envs::AbstractVector{<:Environment3}, accs; λ = :evidence, λself = λ,
        prior = smoothness_prior.(bases), reflections = false, chunksize = 2000, method = :qr, self_fit = :staged)
    check_lengths(envs, accs)
    check_method(method)
    check_pair(bases)
    prior = frame_prior(prior)
    plane, axis = bases
    t_plane, t_ax = frame_targets(plane, envs, accs)
    solve(b, t, k) = solve_targets(b, envs, t; λ = part(λ, k), λself = part(λself, k), prior = prior[k], chunksize, reflections, method,
        self_fit = resolve_self_fit(b, self_fit))
    return FrameModel(plane, solve(plane, t_plane, 1), axis, solve(axis, t_ax, 2))
end

# a value for both parts of a 3D model, or one per part
part(x::NamedTuple, k) = x[(:plane, :axis)[k]]

part(x, k) = x

function frame_targets(plane, envs, accs)
    check_lengths(envs, accs)
    frames = [basis_frame(plane, e) for e in envs]
    t_plane = [complex(dot(a, f[1]), dot(a, f[2])) for (a, f) in zip(accs, frames)]
    t_ax = [complex(dot(a, f[3])) for (a, f) in zip(accs, frames)]
    return t_plane, t_ax
end

"""
    FrameStatistics(plane, axis)

Streamed statistics of the two parts of a 3D model (`design_statistics((plane, axis), envs, accs)`); pool with `+`.
"""
struct FrameStatistics{S}
    plane::S
    axis::S
end

Base.:+(a::FrameStatistics, b::FrameStatistics) = FrameStatistics(a.plane + b.plane, a.axis + b.axis)

restrict_statistics(S::FrameStatistics, cols) = FrameStatistics(select_columns(S.plane, cols[1]), select_columns(S.axis, cols[2]))

function design_statistics(bases::Tuple{FrameBasis, FrameBasis}, envs::AbstractVector{<:Environment3}, accs; chunksize = 2000)
    check_pair(bases)
    t_plane, t_ax = frame_targets(bases[1], envs, accs)
    return FrameStatistics(qr_statistics(bases[1], envs, t_plane; chunksize), qr_statistics(bases[2], envs, t_ax; chunksize))
end

function select_lambda(S::FrameStatistics, Sval::FrameStatistics, bases::Tuple{FrameBasis, FrameBasis}; λs,
        prior = smoothness_prior.(bases), reflections = false, self_fit = :staged)
    prior = frame_prior(prior)
    p = select_lambda(S.plane, Sval.plane, bases[1]; λs, prior = prior[1], reflections, self_fit)
    a = select_lambda(S.axis, Sval.axis, bases[2]; λs, prior = prior[2], reflections, self_fit)
    return (; λ = (plane = p.λ, axis = a.λ), λself = (plane = p.λself, axis = a.λself))
end

function fit(bases::Tuple{FrameBasis, FrameBasis}, S::FrameStatistics; λ = :evidence, λself = λ, prior = smoothness_prior.(bases),
        reflections = false, self_fit = :staged)
    prior = frame_prior(prior)
    solve(b, St, k) = solve_statistics(St, b; λ = part(λ, k), λself = part(λself, k), prior = prior[k], reflections,
        self_fit = resolve_self_fit(b, self_fit))
    return FrameModel(bases[1], solve(bases[1], S.plane, 1), bases[2], solve(bases[2], S.axis, 2))
end

function fit_path(bases::Tuple{FrameBasis, FrameBasis}, envs::AbstractVector{<:Environment3}, accs; λs, λself = nothing,
        prior = smoothness_prior.(bases), reflections = false, chunksize = 2000, method = :qr, self_fit = :staged)
    check_method(method)
    check_pair(bases)
    prior = frame_prior(prior)
    plane, axis = bases
    t_plane, t_ax = frame_targets(plane, envs, accs)
    function path(b, t, γ, k)
        fb = resolve_self_fit(b, self_fit)
        λk = isnothing(λself) ? nothing : part(λself, k)
        if method === :qr
            S = qr_statistics(b, envs, t; chunksize)
            return fb === :joint ? lsq_path(S; λs, prior = γ, reflections) : staged_path(S, b; λs, λself = λk, prior = γ, reflections, self_fit = fb)
        end
        G, h = accumulate_normal_equations(b, envs, t; chunksize)
        return [fb === :joint ? solve_normal_equations(G, h; λ, prior = γ, reflections) :
                staged_normal_equations(G, h, b; λ, λself = something(λk, λ), prior = γ, reflections, self_fit = fb) for λ in λs]
    end
    cp, ca = path(plane, t_plane, prior[1], 1), path(axis, t_ax, prior[2], 2)
    # the λself each part used: per part, joint parts take λ
    used(λ) = NamedTuple{(:plane, :axis)}(Tuple(effective_λself(resolve_self_fit(b, self_fit), isnothing(λself) ? nothing : part(λself, k), λ)
        for (k, b) in enumerate(bases)))
    return [FrameModel(plane, cp[i], axis, ca[i], λs[i], used(λs[i])) for i in eachindex(λs)]
end
