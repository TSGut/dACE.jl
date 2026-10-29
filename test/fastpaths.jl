# the fast paths against independent references: in-place evaluation, cell-list neighbour search, the normal-equation workspace

@testset "reusable evaluation" begin
    rng = Xoshiro(903)
    full = History(selfterm = 1:2, interactions = (own = 1, neighbours = 1), use = :both, selfterm_degree = (3, 2), selfterm_maxm = 1)
    for D in (2, 3), history in (nothing, full), inputmap in (:linear, :tanh)
        X, V = randn(rng, D, 8), randn(rng, D, 8)
        past = [randn(rng, D, 8), randn(rng, D, 8)]
        envs = environments(X, V; cutoff = 5.0, history, past_velocities = past, dt = 0.1)
        kw = (; maxdeg = (2, 3, 3), history, inputmap, normaliser = (1.0, 0.5))
        bases = D == 2 ? (dACEBasis(; kw...),) : (heading_bases(; kw...)..., vertical_bases(; kw...)...)
        for b in bases
            work = EvaluationWorkspace(b)
            out = zeros(ComplexF64, length(b))
            for env in (envs..., environments(X[:, 1:1], V[:, 1:1]; cutoff = 5.0, history, past_velocities = [h[:, 1:1] for h in past], dt = 0.1)...)
                fill!(out, NaN)
                @test evaluate!(out, b, env, work) === out
                @test out ≈ dACE.evaluate_direct(b, env) atol = 1.0e-10
            end
            # Repeated threaded calls own independent scratch, including nested callers.
            expected = reduce(vcat, [transpose(evaluate(b, e)) for e in envs])
            jobs = [Threads.@spawn(dACE.design_matrix(b, envs)) for _ in 1:3]
            @test all(B -> B ≈ expected, fetch.(jobs))
            @test_throws DimensionMismatch evaluate!(ComplexF64[], b, first(envs), work)
        end
    end
end

# Deliberately independent O(N²) reference, retaining original neighbour order.
function reference_environments(pos, vel; cutoff, periodic = false, extent = nothing, past_velocities = nothing, dt = NaN)
    D, N = size(pos)
    E = D == 2 ? Environment : Environment3
    return map(1:N) do i
        r, u, vn = (SVector{D, Float64}[] for _ in 1:3)
        for j in 1:N
            i == j && continue
            dx = SVector{D}(pos[:, j] - pos[:, i])
            periodic && (dx = dx - SVector{D}(extent) .* round.(dx ./ SVector{D}(extent)))
            if norm(dx) < cutoff
                push!(r, dx)
                push!(u, SVector{D}(vel[:, j] - vel[:, i]))
                isnothing(past_velocities) || push!(vn, SVector{D}(past_velocities[1][:, j]))
            end
        end
        v = SVector{D}(vel[:, i])
        E(v, r, u; past_v = isnothing(past_velocities) ? SVector{D, Float64}[] : [SVector{D}(P[:, i]) for P in past_velocities], past_vn = vn, dt)
    end
end

@testset "cell-list environments preserve minimum-image semantics" begin
    rng = Xoshiro(904)
    for D in (2, 3), periodic in (false, true), N in (0, 1, 2, 30), cutoff in (0.0, 0.4, 2.0, 8.0, Inf)
        # Includes coordinates outside the periodic box and cutoffs over half its width.
        X, V, A = 10randn(rng, D, N), randn(rng, D, N), randn(rng, D, N)
        kw = (; cutoff, periodic, extent = ntuple(k -> 3.0 + k, D), past_velocities = [A], dt = 0.1)
        actual = environments(X, V; kw..., history = History(selfterm = 1, interactions = (neighbours = 1,)))
        expected = reference_environments(X, V; kw...)
        @test length(actual) == length(expected)
        for (a, e) in zip(actual, expected)
            @test (a.v, a.r, a.u, a.past_v, a.past_vn, a.dt) == (e.v, e.r, e.u, e.past_v, e.past_vn, e.dt)
        end
    end
    for D in (2, 3), periodic in (false, true), cutoff in (1.0, 2.0, 3.0)
        X = zeros(D, 6)
        # distances exactly at the cutoffs, just inside and outside, and at half the box (no coincident pairs)
        X[1, :] = [0, 0.25, 1, prevfloat(1.0), nextfloat(1.0), 2]
        V = randn(rng, D, 6)
        kw = (; cutoff, periodic, extent = ntuple(_ -> 4.0, D))
        actual, expected = environments(X, V; kw...), reference_environments(X, V; kw...)
        @test getfield.(actual, :r) == getfield.(expected, :r)
        @test getfield.(actual, :u) == getfield.(expected, :u)
    end
    # coincident particles (also periodic images of each other) have no direction
    for D in (2, 3)
        X = zeros(D, 3)
        X[1, :] = [0, 1, 2]
        @test_throws ArgumentError environments(X, randn(rng, D, 3); cutoff = 1.5, periodic = true, extent = ntuple(_ -> 2.0, D))
        @test length(environments(X, randn(rng, D, 3); cutoff = 1.5)) == 3
    end
end

@testset "normal-equation workspace preserves statistics" begin
    rng = Xoshiro(905)
    B = randn(rng, ComplexF64, 40, 9)
    B[:, end] .= 0
    G, h = B' * B, B' * randn(rng, ComplexF64, 40)
    G0, h0 = copy(G), copy(h)
    for reflections in (false, true), prior in (nothing, collect(1.0:9.0)), λ in (1.0e-6, 0.1)
        A, y = reflections ? (real(G), real(h)) : (G, h)
        s = sqrt.(real(diag(A)))
        zero_cols = s .== 0
        s[zero_cols] .= 1
        scaled = A ./ (s * s') + λ^2 * (isnothing(prior) ? I : Diagonal(prior))
        scaled[diagind(scaled)[zero_cols]] .= 1
        expected = complex.((cholesky(Hermitian(scaled)) \ (y ./ s)) ./ s)
        @test dACE.solve_normal_equations(G, h; λ, prior, reflections) ≈ expected
        @test G == G0 && h == h0
    end
end
