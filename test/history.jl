# History: self terms, the agent's own past in the interactions, neighbours' past states

@testset "history" begin
    @testset "History" begin
        @test_throws ArgumentError History(selfterm = 0:1)
        @test_throws ArgumentError History(selfterm = 1, use = :position)
        @test History(selfterm = 2).selfterm == 2:2
        @test_throws ArgumentError History()
        @test_throws ArgumentError History(interactions = (own = 1:2,))
        @test_throws ArgumentError History(interactions = (neighbours = 2,))
        @test_throws ArgumentError History(interactions = (others = 1:1,))
        @test_throws ArgumentError dACEBasis(maxdeg = (2, 3), history = true)
        @test dACE.history_length(History(selfterm = 2:4)) == 4
        @test dACE.history_length(History(interactions = (neighbours = 1:1,))) == 1
        @test dACE.history_length(History(interactions = (own = 1,))) == 1
        @test dACE.history_length(nothing) == 0
        # past states from velocities only: acceleration = difference of consecutive velocities
        e = Environment(SVector(1.0, 0.0), SVector{2, Float64}[], SVector{2, Float64}[]; past_v = [SVector(0.8, 0.1), SVector(0.5, 0.3)], dt = 0.1)
        @test dACE.own_past(e, 1, :acceleration) ≈ (e.v - e.past_v[1]) / 0.1
        @test dACE.own_past(e, 2, :acceleration) ≈ (e.past_v[1] - e.past_v[2]) / 0.1
        @test dACE.own_past(e, 2, :velocity) == e.past_v[2]
        @test_throws ArgumentError evaluate(dACEBasis(maxdeg = (2,), history = History(selfterm = 1:3)), e)
    end

    @testset "self term (temporal body order)" begin
        rngs = Xoshiro(11)
        env = history_env(rngs, random_env(rngs, 4))
        history = History(selfterm = 1:3, use = :both, selfterm_degree = (4, 2), selfterm_maxm = 1)
        plain = dACEBasis(maxdeg = (2, 3, 3), maxl = 1)
        b = dACEBasis(maxdeg = (2, 3, 3), maxl = 1, history = history)
        @test length(b) == length(plain) + length(b.selfterm) && length(b.selfterm) > 0
        @test length(b.selfterm.channels) == 6
        @test all(f -> 1 <= dACE.temporal_order(f) <= 2, b.selfterm.funcs)
        @test maximum(dACE.degree(f) for f in b.selfterm.funcs if dACE.temporal_order(f) == 2) <= 2
        @test dACE.correlation_orders(b) == [dACE.correlation_order.(b.funcs); zeros(Int, length(b.selfterm))]
        # the interaction functions are those of the memoryless basis
        B = evaluate(b, env)
        @test B[1:length(plain)] ≈ evaluate(plain, env)
        @test B ≈ brute_force(b, env) && dACE.evaluate_direct(b, env) ≈ B
        @test evaluate(b, rotate(env, 0.9)) ≈ cis(0.9) .* B
        @test evaluate(b, reflect(env)) ≈ conj.(B)
        @test length(smoothness_prior(b)) == length(b)
        @test all(isfinite, evaluate(b, Environment(zero(env.v), env.r, env.u; past_v = [[zero(env.v)]; env.past_v[2:3]], env.dt)))
        # charge 0 (invariants)
        b0 = dACEBasis(maxdeg = (2, 3), maxl = 1, charge = 0, history = history)
        @test evaluate(b0, rotate(env, 0.9)) ≈ evaluate(b0, env)
        # trajectory_data carries the past velocities (steps without the history are skipped)
        tr = simulate(pair; n_agents = 30, nsteps = 40, seed = 1, extent = (100.0, 100.0), spread = 12.0, speeds = (0.2, 2.5), dt = 0.05)
        e, a = trajectory_data(tr.pos, tr.vel, tr.dt; cutoff = interaction_radius(pair), extent = tr.extent, periodic = true, times = 1:2:39, history = history)
        @test length(e) == 18 * 30
        @test e[1].past_v[2] ≈ tr.vel[:, 1, 3] && e[1].dt == tr.dt && isempty(e[1].past_vn)
        # recovery: simulated forces plus a self term with known real coefficients; a persistence ρ a(t - Δ)
        # is in the span with the linear speed map
        hs, F = Environment{Float64}[], SVector{2, Float64}[]
        for seed in 1:4
            tr = simulate(pair; n_agents = 30, nsteps = 40, seed, extent = (100.0, 100.0), spread = 12.0, speeds = (0.2, 2.5), dt = 0.05)
            x, y = trajectory_data(tr.pos, tr.vel, tr.dt; cutoff = interaction_radius(pair), extent = tr.extent, periodic = true, times = 5:2:39, history = history)
            append!(hs, x); append!(F, y)
        end
        ib = dACEBasis(maxdeg = (4, 8), maxl = 0, maxm = 1)
        bs = dACEBasis(maxdeg = (4, 8), maxl = 0, maxm = 1, history = history)
        cs = zeros(length(bs.selfterm))
        cs[rand(rngs, 1:length(cs), 6)] .= randn(rngs, 6)
        selfpart(x) = (w = sum(cs .* evaluate(bs, x)[(length(ib) + 1):end]); SVector(real(w), imag(w)))
        g = selfpart.(hs)
        A = F .+ 0.3 .* g .+ 0.5 .* [dACE.own_past(x, 1, :acceleration) for x in hs]
        m = fit(bs, hs, A; λ = 1.0e-8, reflections = true, self_fit = :joint)
        @test rmse(m, hs, A; relative = true) < 1.0e-2
        # staged (default): the interactions are those of the fit without the self term, and the self term is the
        # ridge fit of their residual; the normal equations, the path and CompositeBasis agree
        γ = smoothness_prior(bs)
        K1 = length(ib)
        ms = fit(bs, hs, A; λ = 1.0e-3, λself = 1.0e-4, prior = γ, reflections = true)
        m0 = fit(ib, hs, A; λ = 1.0e-3, prior = γ[1:K1], reflections = true)
        @test ms.c[1:K1] ≈ m0.c
        res = A .- predict(m0, hs)
        Bs = reduce(vcat, transpose(evaluate(bs, x)[(K1 + 1):end]) for x in hs)
        cself = dACE.ridge_solve([real(Bs); imag(Bs)], [first.(res); last.(res)]; λ = 1.0e-4, prior = γ[(K1 + 1):end])
        @test ms.c[(K1 + 1):end] ≈ cself rtol = 1.0e-6
        @test fit(bs, hs, A; λ = 1.0e-3, λself = 1.0e-4, prior = γ, reflections = true, method = :normal_equations).c ≈ ms.c rtol = 1.0e-6
        @test fit_path(bs, hs, A; λs = [1.0e-2, 1.0e-3], λself = 1.0e-4, prior = γ, reflections = true)[2].c ≈ ms.c
        @test rmse(ms, hs, A; relative = true) < rmse(m0, hs, A; relative = true)
        @test fit(bs, hs, A; λ = 1.0e-3, prior = γ, reflections = true, self_fit = :joint).c[1:K1] ≉ m0.c
        @test fit(ib, hs, A; λ = 1.0e-3, reflections = true, self_fit = :joint).c ≈ fit(ib, hs, A; λ = 1.0e-3, reflections = true).c
        S = dACE.qr_statistics(bs, hs, [complex(a...) for a in A])
        Sv = dACE.qr_statistics(bs, hs[1:200], [complex(a...) for a in A[1:200]])
        λ1, λ2 = dACE.select_staged(S, Sv, bs; λs = [1.0e-1, 1.0e-3, 1.0e-6], prior = γ, reflections = true)
        @test λ1 in (1.0e-1, 1.0e-3, 1.0e-6) && λ2 in (1.0e-1, 1.0e-3, 1.0e-6)
        # self first: the self term alone on the targets, then the interactions on its residual
        mr = fit(bs, hs, A; λ = 1.0e-3, λself = 1.0e-4, prior = γ, reflections = true, self_fit = :self_first)
        realsys(B) = [real(B); imag(B)]
        cs1 = dACE.ridge_solve(realsys(Bs), [first.(A); last.(A)]; λ = 1.0e-4, prior = γ[(K1 + 1):end])
        @test mr.c[(K1 + 1):end] ≈ cs1 rtol = 1.0e-6
        rs = A .- [SVector(real(w), imag(w)) for w in Bs * cs1]
        Bi = reduce(vcat, transpose(evaluate(ib, x)) for x in hs)
        @test mr.c[1:K1] ≈ dACE.ridge_solve(realsys(Bi), [first.(rs); last.(rs)]; λ = 1.0e-3, prior = γ[1:K1]) rtol = 1.0e-6
        @test fit(bs, hs, A; λ = 1.0e-3, λself = 1.0e-4, prior = γ, reflections = true, self_fit = :self_first, method = :normal_equations).c ≈ mr.c rtol = 1.0e-6
        @test fit_path(bs, hs, A; λs = [1.0e-2, 1.0e-3], λself = 1.0e-4, prior = γ, reflections = true, self_fit = :self_first)[2].c ≈ mr.c
        @test all(in((1.0e-1, 1.0e-3, 1.0e-6)), dACE.select_staged(S, Sv, bs; λs = [1.0e-1, 1.0e-3, 1.0e-6], prior = γ, reflections = true, self_fit = :self_first))
        @test_throws ArgumentError fit(bs, hs, A; self_fit = :both)
        @test_throws ArgumentError dACE.staged_solve(S, bs; self_fit = :joint)
        @test predict(m, hs) ≈ [predict(m, x) for x in hs]
        @test all(sum(body_order_contributions(m, x)) ≈ predict(m, x) for x in hs[1:50])
        @test rmse(fit(ib, hs, A; λ = 1.0e-8, reflections = true), hs, A; relative = true) > 5rmse(m, hs, A; relative = true)
        # rollouts carry the velocity history
        P, V = rollout(m, tr.pos[:, 1:10, 1], tr.vel[:, 1:10, 1], tr.dt, 5; extent = tr.extent, periodic = true)
        @test size(P) == (2, 10, 6) && all(isfinite, P)
        @test dACE.has_history(m) && dACE.history_of(m) === history
        # 3D, both frames: equivariance, the purified and direct evaluations agree
        rng3 = Xoshiro(5)
        env3 = history_env3(rng3, Environment3(SVector(0.3, -1.1, 0.4),
            SVector{3, Float64}[(0.4 + 4rand(rng3)) * normalize(randn(rng3, SVector{3})) for _ in 1:4],
            SVector{3, Float64}[0.5randn(rng3, SVector{3}) for _ in 1:4]))
        Rz = SMatrix{3, 3}(cos(0.7), sin(0.7), 0, -sin(0.7), cos(0.7), 0, 0, 0, 1)
        Q = SMatrix{3, 3}(Matrix(qr(randn(rng3, 3, 3)).Q))
        Q = det(Q) < 0 ? -Q : Q
        for (bases, R3) in ((heading_bases(maxdeg = (2, 3, 3), history = history), Q), (vertical_bases(maxdeg = (2, 3, 3), history = history, inputmap = :tanh), Rz))
            @test all(b -> length(b.selfterm) > 0 && evaluate(b, env3) ≈ brute_force(b, env3) && dACE.evaluate_direct(b, env3) ≈ evaluate(b, env3), bases)
            m3 = FrameModel(bases[1], randn(rng3, ComplexF64, length(bases[1])), bases[2], randn(rng3, ComplexF64, length(bases[2])))
            @test dACE.has_history(m3)
            @test predict(m3, rotate(env3, R3)) ≈ R3 * predict(m3, env3)
            @test sum(body_order_contributions(m3, env3)) ≈ predict(m3, env3)
        end
        # a self model of one's own through CompositeBasis
        cb = CompositeBasis(ib, dACEBasis(maxdeg = (2,), history = history))
        @test length(cb) == length(ib) + length(dACEBasis(maxdeg = (2,), history = history))
        @test evaluate(cb, hs[1]) ≈ [evaluate(ib, hs[1]); evaluate(cb.selfterm, hs[1])]
        mc = fit(cb, hs, A; λ = 1.0e-8, reflections = true)
        @test mc isa dACEModel{<:CompositeBasis} && predict(mc, hs[1:5]) ≈ [predict(mc, x) for x in hs[1:5]]
        # the two parts must agree on the past they need
        @test_throws ArgumentError dACE.basis_history(CompositeBasis(dACEBasis(maxdeg = (2,), history = History(selfterm = 1)), dACEBasis(maxdeg = (2,), history = History(selfterm = 2))))
    end

    @testset "interaction history (the agent's own past in the reactions to neighbours)" begin
        rng4 = Xoshiro(4)
        env = history_env(rng4, random_env(rng4, 5))
        history = History(selfterm = 1, interactions = (own = 1,), use = :acceleration, interaction_degree = 2, selfterm_degree = (2, 0))
        plain = dACEBasis(maxdeg = (2, 3, 3), maxl = 1)
        bm = dACEBasis(maxdeg = (2, 3, 3), maxl = 1, history = history, ascale = 2.0)
        # the agent's last state enters the interaction terms; its 1-body dependence is the self term
        @test length(bm.funcs) > length(plain.funcs)
        @test all(f -> dACE.correlation_order(f) >= 1 || (bm.centres[f.centre].k, bm.centres[f.centre].q) == (0, 0), bm.funcs)
        @test any(f -> dACE.correlation_order(f) >= 1 && bm.centres[f.centre].k != 0, bm.funcs)
        @test dACE.own_scaled(bm, env) ≈ (env.v - env.past_v[1]) / env.dt / 2.0
        B = evaluate(bm, env)
        @test B ≈ brute_force(bm, env) && dACE.evaluate_direct(bm, env) ≈ B
        for α in (0.3, 1.7)
            @test evaluate(bm, rotate(env, α)) ≈ cis(α) * B
        end
        @test evaluate(bm, reflect(env)) ≈ conj.(B)
        bv = dACEBasis(maxdeg = (2, 3, 3), maxl = 1, history = History(selfterm = 1, interactions = (own = 1,)))
        @test dACE.own_scaled(bv, env) ≈ env.past_v[1] / bv.vscale
        # a pair force whose strength depends on the agent's past acceleration: learned only with own (staged:
        # the plain interactions, the self term, then the own terms)
        envs, F = Environment{Float64}[], SVector{2, Float64}[]
        for seed in 1:4
            tr = simulate(pair; n_agents = 30, nsteps = 60, seed, extent = (100.0, 100.0), spread = 12.0, speeds = (0.2, 2.5), dt = 0.05)
            x, y = trajectory_data(tr.pos, tr.vel, tr.dt; cutoff = interaction_radius(pair), extent = tr.extent, periodic = true, times = 2:2:60, history = history)
            append!(envs, x); append!(F, y)
        end
        R = interaction_radius(pair)
        modulated(x) = (a = dACE.own_past(x, 1, :acceleration) / 2.0; sum((dACE.fcut(norm(r), R) * (complex(a[1], a[2]) * conj(complex(r[1], r[2])) / norm(r) |> real) * r / norm(r) for r in x.r); init = zero(x.v)))
        targets = F .+ modulated.(envs)
        kw = (maxdeg = (4, 6), maxl = 0, maxm = 2, cutoff = R, ascale = 2.0)
        with = fit(dACEBasis(; kw..., history = history), envs, targets)
        selfonly = fit(dACEBasis(; kw..., history = History(selfterm = 1, use = :acceleration, selfterm_degree = (2, 0))), envs, targets)
        joint = fit(dACEBasis(; kw..., history = history), envs, targets; self_fit = :joint)
        @test rmse(joint, envs, targets; relative = true) < 0.5rmse(selfonly, envs, targets; relative = true)
        # staged (the own terms last, on the residual) recovers part of it: the price of keeping the plain interactions
        @test rmse(with, envs, targets; relative = true) < rmse(selfonly, envs, targets; relative = true)
        # three stages: the plain interactions are those of the fit without any past, the own terms come last
        bw = dACEBasis(; kw..., history = history)
        P, O, Sc = dACE.column_blocks(bw)
        @test !isempty(O) && !isempty(Sc) && sort([P; O; Sc]) == 1:length(bw)
        @test with.c[P] ≈ fit(dACEBasis(; kw...), envs, targets).c
        @test fit(bw, envs, targets; λ = with.λ, λself = with.λself, method = :normal_equations).c ≈ with.c rtol = 1.0e-5
        @test fit_path(bw, envs, targets; λs = [1.0e-3, with.λ], λself = with.λself)[end].c ≈ with.c
        # validation-based selection: the own stage reuses the interactions' λ
        Sw, Svw = design_statistics(bw, envs, targets), design_statistics(bw, envs[1:300], targets[1:300])
        @test all(in([1.0e-2, 1.0e-4]), dACE.select_staged(Sw, Svw, bw; λs = [1.0e-2, 1.0e-4]))
    end

    @testset "neighbour history (neighbours' past states)" begin
        rng7 = Xoshiro(8)
        e0 = random_env(rng7, 5)
        env = history_env(rng7, e0; H = 1)
        @test_throws DimensionMismatch Environment(e0.v, e0.r, e0.u; past_v = env.past_v, past_vn = [randn(rng7, SVector{2})], dt = 0.1)
        history = History(interactions = (neighbours = 1,), use = :acceleration, interaction_degree = 2)
        bn = dACEBasis(maxdeg = (2, 3, 3), maxl = 1, history = history)
        @test length(bn) > length(dACEBasis(maxdeg = (2, 3, 3), maxl = 1)) && isnothing(bn.selfterm)
        B = evaluate(bn, env)
        @test B ≈ brute_force(bn, env) && dACE.evaluate_direct(bn, env) ≈ B
        @test evaluate(bn, rotate(env, 0.4)) ≈ cis(0.4) * B
        @test evaluate(bn, reflect(env)) ≈ conj.(B)
        p = randperm(rng7, 5)
        @test evaluate(bn, Environment(env.v, env.r[p], env.u[p]; past_v = env.past_v, past_vn = env.past_vn[p], env.dt)) ≈ B
        @test dACE.neighbour_past(env, 2, :acceleration) ≈ (env.v + env.u[2] - env.past_vn[2]) / env.dt
        @test dACE.neighbour_past(env, 2, :velocity) ≈ env.past_vn[2] - env.past_v[1]
        @test_throws ArgumentError evaluate(bn, Environment(env.v, env.r, env.u; past_v = env.past_v, dt = 0.1))
        # environments and trajectory_data carry the neighbours' previous velocities
        tr = simulate(pair; n_agents = 30, nsteps = 60, seed = 1, extent = (100.0, 100.0), spread = 12.0, speeds = (0.2, 2.5), dt = 0.05)
        e, _ = trajectory_data(tr.pos, tr.vel, tr.dt; cutoff = interaction_radius(pair), extent = tr.extent, periodic = true, times = 1:2:60, history = history)
        @test all(x -> length(x.past_vn) == length(x.r) && length(x.past_v) == 1, e)
        es = environments(tr.pos[:, :, 5], tr.vel[:, :, 5]; cutoff = 30.0, history, past_velocities = [tr.vel[:, :, 4]], dt = tr.dt)
        i = findfirst(x -> length(x) > 0, es)
        j = findfirst(k -> k != i && norm(tr.pos[:, k, 5] - tr.pos[:, i, 5]) < 30, 1:30)
        @test es[i].past_vn[1] ≈ tr.vel[:, j, 4] && es[i].past_v[1] ≈ tr.vel[:, i, 4]
        # a = pair force + Σⱼ fc(ρⱼ) bⱼ, bⱼ the neighbour's past acceleration (independent of the state): learned
        # only with neighbour history
        envs, accs = swarm_data(pair, 1:4)
        R = interaction_radius(pair)
        envs = [Environment(x.v, x.r, x.u; past_v = [x.v], past_vn = [x.v + u + 0.1randn(rng7, SVector{2}) for u in x.u], dt = 0.1) for x in envs]
        targets = accs .+ [sum((dACE.fcut(norm(r), R) * dACE.neighbour_past(x, j, :acceleration) for (j, r) in enumerate(x.r)); init = zero(x.v)) for x in envs]
        with = fit(dACEBasis(maxdeg = (4, 6), cutoff = R, history = History(interactions = (neighbours = 1,), use = :acceleration)), envs, targets)
        @test rmse(with, envs, targets; relative = true) < 0.02
        @test dACE.has_history(with)
        P, _ = rollout(with, tr.pos[:, 1:10, 1], tr.vel[:, 1:10, 1], tr.dt, 5; extent = tr.extent, periodic = true)
        @test size(P) == (2, 10, 6) && all(isfinite, P)
        # 3D: both frames, with the agent's history in the interactions too
        rng8 = Xoshiro(9)
        env3 = history_env3(rng8, Environment3(SVector(0.3, -1.1, 0.4),
            SVector{3, Float64}[(0.4 + 4rand(rng8)) * normalize(randn(rng8, SVector{3})) for _ in 1:4],
            SVector{3, Float64}[0.5randn(rng8, SVector{3}) for _ in 1:4]); H = 1)
        Rz = SMatrix{3, 3}(cos(0.7), sin(0.7), 0, -sin(0.7), cos(0.7), 0, 0, 0, 1)
        Q = SMatrix{3, 3}(Matrix(qr(randn(rng8, 3, 3)).Q))
        Q = det(Q) < 0 ? -Q : Q
        m3d = History(selfterm = 1, interactions = (own = 1, neighbours = 1), interaction_degree = 2, selfterm_degree = (2, 0))
        for (bases, R3) in ((heading_bases(maxdeg = (2, 3, 3), history = m3d), Q), (vertical_bases(maxdeg = (2, 3, 3), history = m3d, inputmap = :tanh), Rz))
            for b in bases
                @test evaluate(b, env3) ≈ brute_force(b, env3)
                @test dACE.evaluate_direct(b, env3) ≈ evaluate(b, env3)
            end
            m3 = FrameModel(bases[1], randn(rng8, ComplexF64, length(bases[1])), bases[2], randn(rng8, ComplexF64, length(bases[2])))
            @test dACE.has_history(m3)
            @test predict(m3, rotate(env3, R3)) ≈ R3 * predict(m3, env3)
        end
    end
end
