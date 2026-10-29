# the normal-equations route (method = :normal_equations), cheaper to accumulate, squares the condition number

function accumulate_normal_equations(basis, envs, targets; chunksize = 2000)
    check_lengths(envs, targets)
    K = length(basis)
    G = zeros(ComplexF64, K, K)
    h = zeros(ComplexF64, K)
    works = workspace_pool(basis)
    for chunk in Iterators.partition(eachindex(envs), chunksize)
        B = design_matrix(basis, view(envs, chunk); works)
        mul!(G, B', B, true, true)
        mul!(h, B', view(targets, chunk), true, true)
    end
    return G, h
end

function solve_normal_equations(G, h; λ = 1.0e-6, prior = nothing, reflections = false)
    check_lambda(λ)
    isnothing(prior) || prior_weights(prior, length(h))
    s = sqrt.(real(diag(G)))
    zero_cols = s .== 0  # functions that vanish on all data: coefficient 0
    s[zero_cols] .= 1
    T = reflections ? eltype(s) : promote_type(eltype(G), eltype(s))
    G̃ = Matrix{T}(undef, size(G))
    for j in axes(G, 2), i in axes(G, 1)
        G̃[i, j] = (reflections ? real(G[i, j]) : G[i, j]) / (s[i] * s[j])
    end
    for i in eachindex(s)
        G̃[i, i] += λ^2 * (isnothing(prior) ? 1 : prior[i])
        zero_cols[i] && (G̃[i, i] = 1)
    end
    rhs = reflections ? real.(h) ./ s : h ./ s
    ldiv!(cholesky!(Hermitian(G̃)), rhs)
    return complex.(rhs ./ s)
end

normal_equations_fit(basis, envs, targets; λ = 1.0e-6, prior = nothing, chunksize = 2000, reflections = false) =
    solve_normal_equations(accumulate_normal_equations(basis, envs, targets; chunksize)...; λ, prior, reflections)

# the stages of `staged_solve` on accumulated normal equations
function staged_normal_equations(G, h, basis; λ, λself = λ, prior = nothing, reflections = false, self_fit = :staged)
    c = zeros(ComplexF64, length(h))
    done = Int[]
    for (cols, tag) in fit_stages(basis, self_fit)
        isempty(cols) && continue
        c[cols] .= solve_normal_equations(G[cols, cols], h[cols] - G[cols, done] * c[done];
            λ = stage_lambda(tag, λ, λself), prior = stage_prior(prior, cols), reflections)
        append!(done, cols)
    end
    return c
end
