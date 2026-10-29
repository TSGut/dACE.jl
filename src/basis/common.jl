# pieces shared by the 2D and 3D bases: complex powers, input maps, cutoffs, set partitions, degree tuples

# z = x + iy; rotation z ↦ e^{iα} z; output aₓ + i a_y has charge 1

cpow(z, m) = m >= 0 ? z^m : conj(z)^(-m)

const INPUTMAPS = (:linear, :tanh)

"""
    scaled(x, s, inputmap)

`x / s` (`:linear`) or `tanh(|x| / s) x / |x|` (`:tanh`, magnitudes in `[0, 1)`): the scaled
velocity or acceleration seen by the basis; directions are kept, so both maps commute with
rotations and reflections.
"""
scaled(x, s, inputmap) = inputmap === :linear ? x / s : (n = norm(x); n == 0 ? zero(x) : tanh(n / s) * x / n)

"""
    set_partitions(N)

Partitions of `1:N`.
"""
function set_partitions(N)
    N == 0 && return [Vector{Int}[]]
    out = Vector{Vector{Vector{Int}}}()
    for p in set_partitions(N - 1)
        for b in eachindex(p)
            q = deepcopy(p)
            push!(q[b], N)
            push!(out, q)
        end
        push!(out, [deepcopy(p); [[N]]])
    end
    return out
end

# Möbius function of the partition lattice
mobius(p) = prod(B -> (-1)^(length(B) - 1) * factorial(length(B) - 1), p; init = 1)

# non-decreasing N-tuples with total degree ≤ D (`degs` sorted)
function sorted_tuples(degs, N, D)
    out = Vector{Vector{Int}}()
    function rec(t, start, d)
        length(t) == N && return push!(out, copy(t))
        for k in start:length(degs)
            d + degs[k] > D && break
            push!(t, k)
            rec(t, k, d + degs[k])
            pop!(t)
        end
        return nothing
    end
    rec(Int[], 1, 0)
    return out
end

fcut(ρ, R) = ρ < R ? (1 - (ρ / R)^2)^2 : zero(ρ)

function check_normaliser(x)
    isnothing(x) && return nothing
    c, p = Float64.(x)
    c > 0 && p > 0 || throw(ArgumentError("normaliser = (c, p) needs c > 0 and p > 0"))
    return (c, p)
end

"""
    interaction_weight(basis, env)

`(c + S₀)^(-p)`, `S₀ = Σⱼ fc(ρⱼ)`, for `normaliser = (c, p)`; 1 without a normaliser.
"""
function interaction_weight(basis, env)
    isnothing(basis.normaliser) && return 1.0
    c, p = basis.normaliser
    return (c + sum((fcut(norm(r), basis.cutoff) for r in env.r); init = 0.0))^(-p)
end

basis_history(b) = hasproperty(b, :history) ? b.history : nothing
