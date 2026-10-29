# evaluation of the bases: purity, symmetries, contributions, envelopes, speed maps, normalisers

@testset "basis" begin
    rng = Xoshiro(1)
    basis = dACEBasis(maxdeg = (2, 3, 3, 2), maxl = 1)
    env = random_env(rng, 6)

    @testset "pure products have no self-interactions" begin
        @test maximum(dACE.correlation_order, basis.funcs) == 3
        @test evaluate(basis, env) ≈ brute_force(basis, env)
        @test dACE.evaluate_direct(basis, env) ≈ brute_force(basis, env)
        for b in (
                dACEBasis(maxdeg = (2, 3, 3, 2), maxl = 1, maxm = 2), dACEBasis(maxdeg = (4, 8, 6)),
                dACEBasis(maxdeg = (4, 8), maxl = 0, maxm = 1), socialforce_basis(),
                dACEBasis(maxdeg = (2, 3, 3, 2), maxl = 1, inner_envelope = 2),
            )
            @test evaluate(b, env) ≈ dACE.evaluate_direct(b, env)
        end
    end

    @testset "self-interacting (unpurified) basis" begin
        si = dACEBasis(maxdeg = (2, 3, 3, 2), maxl = 1, pure = false)
        Φ = dACE.onep_values(si, env)
        Ψ = dACE.centre_values(si, env)
        A = vec(sum(Φ; dims = 1))
        @test evaluate(si, env) ≈ [Ψ[f.centre] * prod(A[f.ks]; init = one(ComplexF64)) for f in si.funcs]
        @test length(si) == length(basis)
        @test !(evaluate(si, env) ≈ evaluate(basis, env))
    end

    @testset "symmetries" begin
        B = evaluate(basis, env)
        for α in (0.3, 1.7, π)
            @test evaluate(basis, rotate(env, α)) ≈ cis(α) * B
        end
        p = randperm(rng, length(env))
        @test evaluate(basis, Environment(env.v, env.r[p], env.u[p])) ≈ B
        @test evaluate(basis, Environment(env.v, SVector{2, Float64}[], SVector{2, Float64}[])) ≈
            [dACE.correlation_order(f) == 0 ? b : 0 for (f, b) in zip(basis.funcs, B)]
    end

    @testset "body-order contributions sum to the prediction" begin
        m = dACEModel(basis, randn(rng, ComplexF64, length(basis)))
        @test sum(body_order_contributions(m, env)) ≈ predict(m, env)
        @test predict(m, rotate(env, 0.9)) ≈ rotation(0.9) * predict(m, env)
    end

    @testset "reflections" begin
        σ(x) = SVector(x[1], -x[2])
        @test evaluate(basis, reflect(env)) ≈ conj.(evaluate(basis, env))
        m_real = dACEModel(basis, complex.(randn(rng, length(basis))))
        @test predict(m_real, reflect(env)) ≈ σ(predict(m_real, env))
        m_cplx = dACEModel(basis, randn(rng, ComplexF64, length(basis)))
        @test !(predict(m_cplx, reflect(env)) ≈ σ(predict(m_cplx, env)))
    end

    @testset "inner envelope and smoothness prior" begin
        b = dACEBasis(maxdeg = (2, 3, 3), maxl = 1, inner_envelope = 2)
        @test brute_force(b, env) ≈ evaluate(b, env)
        near = dACE.Environment(env.v, [[1.0e-9 * env.r[1]]; env.r[2:end]], env.u)
        far = dACE.Environment(env.v, env.r[2:end], env.u[2:end])
        @test evaluate(b, near) ≈ evaluate(b, far) atol = 1.0e-6
        envs, accs = swarm_data(pair, 1:2)
        b = dACEBasis(maxdeg = (4, 6))
        γ = smoothness_prior(b)
        @test length(γ) == length(b) && all(>=(1), γ)
        m = fit(b, envs, accs; λ = 1.0e-2, prior = γ)
        @test fit(b, envs, accs; λ = 1.0e-2, prior = ones(length(b))).c ≈ fit(b, envs, accs; λ = 1.0e-2, prior = nothing).c
        # the default prior is the smoothness prior
        @test fit(b, envs, accs; λ = 1.0e-2).c ≈ m.c
    end

    @testset "speed map" begin
        rng6 = Xoshiro(7)
        e0 = random_env(rng6, 5)
        env = history_env(rng6, Environment(3 * e0.v, e0.r, 3 .* e0.u))
        history = History(selfterm = 1, interactions = (own = 1,), use = :acceleration)
        @test dACE.scaled(SVector(3.0, 4.0), 2.0, :linear) ≈ SVector(1.5, 2.0)
        @test norm(dACE.scaled(SVector(3.0, 4.0), 2.0, :tanh)) ≈ tanh(2.5)
        @test dACE.scaled(SVector(0.0, 0.0), 2.0, :tanh) == SVector(0.0, 0.0)
        @test_throws ArgumentError dACEBasis(maxdeg = (2, 3), inputmap = :log)
        lin = dACEBasis(maxdeg = (2, 3, 3), maxl = 1, history = history)
        @test evaluate(dACEBasis(maxdeg = (2, 3, 3), maxl = 1, history = history, inputmap = :linear), env) ≈ evaluate(lin, env)
        bt = dACEBasis(maxdeg = (2, 3, 3), maxl = 1, history = history, inputmap = :tanh)
        B = evaluate(bt, env)
        @test B ≈ brute_force(bt, env)
        @test !(B ≈ evaluate(lin, env))
        @test evaluate(bt, rotate(env, 1.1)) ≈ cis(1.1) * B
        @test evaluate(bt, reflect(env)) ≈ conj.(B)
        # bounded: a 1-body tanh basis stays finite and saturates for fast agents
        b1 = dACEBasis(maxdeg = (6,), inputmap = :tanh)
        fast(s) = evaluate(b1, Environment(SVector(s, 0.0), SVector{2, Float64}[], SVector{2, Float64}[]))
        @test fast(1.0e3) ≈ fast(1.0e4)
        hb = heading_bases(maxdeg = (2, 3, 3), maxl = 1, inputmap = :tanh)
        env3 = Environment3(SVector(2.0, -1.0, 3.0), SVector{3, Float64}[SVector(1.0, 0.5, -0.3), SVector(-0.4, 2.0, 1.0)],
            SVector{3, Float64}[SVector(3.0, 0.0, 1.0), SVector(-1.0, 4.0, 0.0)])
        @test all(b -> evaluate(b, env3) ≈ brute_force(b, env3), hb)
    end

    @testset "invariant (charge-0) basis" begin
        rng9 = Xoshiro(10)
        env = random_env(rng9, 5)
        b0 = dACEBasis(maxdeg = (2, 3, 3), maxl = 1, charge = 0)
        @test all(f -> dACE.charge(b0.centres[f.centre]) + sum(k -> dACE.charge(b0.onep[k]), f.ks; init = 0) == 0, b0.funcs)
        B = evaluate(b0, env)
        @test B ≈ brute_force(b0, env)
        @test evaluate(b0, rotate(env, 0.9)) ≈ B
        @test evaluate(b0, reflect(env)) ≈ conj.(B)
    end

    @testset "fixed normaliser" begin
        rng10 = Xoshiro(11)
        env = random_env(rng10, 6)
        plain = dACEBasis(maxdeg = (2, 3, 3), maxl = 1)
        normed = dACEBasis(maxdeg = (2, 3, 3), maxl = 1, normaliser = (1.0, 0.5))
        @test_throws ArgumentError dACEBasis(maxdeg = (2, 3), normaliser = (0.0, 1.0))
        S0 = sum(dACE.fcut(norm(r), plain.cutoff) for r in env.r)
        w = [dACE.correlation_order(f) == 0 ? 1.0 : (1 + S0)^(-0.5) for f in plain.funcs]
        @test evaluate(normed, env) ≈ evaluate(plain, env) .* w
        @test dACE.evaluate_direct(normed, env) ≈ evaluate(normed, env)
        @test evaluate(normed, rotate(env, 0.6)) ≈ cis(0.6) * evaluate(normed, env)
        # a normalised pairwise rule is recovered with the normaliser
        envs, _ = swarm_data(pair, 1:3)
        bn = dACEBasis(maxdeg = (2, 4), maxl = 0, maxm = 1, cutoff = interaction_radius(pair), normaliser = (1.0, 1.0))
        truth = dACEModel(bn, complex.(randn(rng10, length(bn))))
        targets = predict(truth, envs)
        @test rmse(fit(bn, envs, targets; λ = 1.0e-8), envs, targets; relative = true) < 1.0e-4
        hb = heading_bases(maxdeg = (2, 3, 3), normaliser = (2.0, 1.0))
        hp = heading_bases(maxdeg = (2, 3, 3))
        env3 = Environment3(SVector(0.3, -1.1, 0.7), SVector{3, Float64}[SVector(1.0, 0.5, -0.3), SVector(-0.4, 2.0, 1.0)],
            SVector{3, Float64}[SVector(0.3, 0.0, 0.1), SVector(-0.1, 0.4, 0.0)])
        S3 = sum(dACE.fcut(norm(r), hb[1].cutoff) for r in env3.r)
        @test evaluate(hb[1], env3) ≈ evaluate(hp[1], env3) .* [dACE.correlation_order(f) == 0 ? 1.0 : 1 / (2 + S3) for f in hp[1].funcs]
    end
end
