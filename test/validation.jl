# errors for invalid input, and regressions of fixed defects

@testset "validation" begin
    @testset "coincident neighbours and non-finite data are errors" begin
        v = SVector(1.0, 0.2)
        @test_throws ArgumentError Environment(v, [SVector(1.0, 0.0), SVector(0.0, 0.0)], [zero(v), zero(v)])
        @test_throws ArgumentError Environment3(SVector(1.0, 0.2, 0.0), [zero(SVector{3, Float64})], [zero(SVector{3, Float64})])
        @test length(Environment(v, [SVector(1.0e-12, 0.0)], [zero(v)])) == 1
        W = randn(Xoshiro(12), 50, 4)
        W[17, 2] = NaN
        @test_throws ArgumentError dACE.qr_statistics(W, 3)
        b = dACEBasis(maxdeg = (4, 6))
        envs = [Environment(SVector(1.0, 0.3), [SVector(0.5, 0.1)], [SVector(0.0, 0.2)]) for _ in 1:20]
        accs = [SVector(0.1, 0.0) for _ in 1:20]
        accs[5] = SVector(Inf, 0.0)
        @test_throws ArgumentError fit(b, envs, accs; λ = 1.0e-3)
    end

    @testset "regressions: fitting, history and 3D parts" begin
        envs, accs = swarm_data(pair, 1:2)
        bp = dACEBasis(maxdeg = (4, 6), maxl = 0, maxm = 1, cutoff = interaction_radius(pair))
        envs3, accs3 = swarm_data3(threebody, 1:2)
        bs = heading_bases(maxdeg = (4, 6), maxl = 0, maxm = 1, maxa = 1, maxb = 1, cutoff = interaction_radius(threebody))
        h = History(selfterm = 1:2, selfterm_degree = (4, 2), selfterm_maxm = 1)
        sims = [simulate(pair; n_agents = 30, nsteps = 40, seed, extent = (100.0, 100.0), spread = 12.0, speeds = (0.2, 2.5), dt = 0.05) for seed in 1:2]
        hd = [trajectory_data(t.pos, t.vel, t.dt; cutoff = interaction_radius(pair), extent = t.extent, periodic = true, times = 3:2:39, history = h) for t in sims]
        henvs, haccs = reduce(vcat, first.(hd)), reduce(vcat, last.(hd))
        bh = dACEBasis(maxdeg = (4, 6), maxl = 0, maxm = 1, cutoff = interaction_radius(pair), history = h)

        # a fixed interaction λ with an evidence λself: the self stage is chosen on the residual of that fit
        S = design_statistics(bh, henvs, haccs)
        γ = smoothness_prior(bh)
        m = fit(bh, S; λ = 1.0e-3, λself = :evidence, reflections = true)
        F, K = dACE.factor(S, true), S.K
        plain, _, selfcols = dACE.column_blocks(bh)
        ci = dACE.ridge_solve(view(F, :, plain), F[:, K + 1]; λ = 1.0e-3, prior = γ[plain])
        r = F[:, K + 1] - view(F, :, plain) * real(ci)
        λs_manual, cs_manual = dACE.evidence_solve(view(F, :, selfcols), r, 2S.n; prior = γ[selfcols])
        @test m.λ == 1.0e-3 && m.λself == λs_manual
        @test m.c[plain] ≈ ci
        @test m.c[selfcols] ≈ cs_manual rtol = 1.0e-8
        @test fit(bh, S; λ = :evidence, λself = 1.0e-4, reflections = true).λself == 1.0e-4

        # mismatched, empty and all-zero data are errors, not truncated or NaN
        mp = fit(bp, envs, accs; λ = 1.0e-3)
        @test_throws DimensionMismatch rmse(mp, envs[1:2], accs[1:3])
        @test_throws ArgumentError rmse(mp, envs[1:0], accs[1:0])
        @test_throws ArgumentError rmse(mp, envs[1:2], [zero(accs[1]) for _ in 1:2]; relative = true)
        @test_throws DimensionMismatch design_statistics(bs, envs3[1:2], accs3[1:3])
        @test_throws DimensionMismatch fit_path(bp, envs[1:2], accs[1:3]; λs = [1.0e-3], method = :normal_equations)

        # a direct solver honours reflections and owns its regularisation
        md = fit(bp, envs, accs; solver = dACE.ACEfit.QR(lambda = 1.0e-8), reflections = true)
        @test all(iszero, imag.(md.c))
        @test relrms(predict(md, envs), predict(fit(bp, envs, accs; λ = 1.0e-6, reflections = true), envs)) < 1.0e-6
        @test_throws ArgumentError fit(bp, envs, accs; solver = dACE.ACEfit.QR(lambda = 1.0e-8), λ = 1.0e-3)

        # λ and prior weights must be positive and finite
        for λ in (0.0, Inf, -1.0)
            @test_throws ArgumentError fit(bp, envs, accs; λ)
            @test_throws ArgumentError fit_path(bp, envs, accs; λs = [1.0e-3, λ])
        end
        for w in (0.0, NaN)
            @test_throws ArgumentError fit(bp, envs, accs; λ = 1.0e-3, prior = [w; ones(length(bp) - 1)])
        end

        # self-history features are continuous and equivariant where a past state vanishes
        hz = History(selfterm = 1, selfterm_degree = (2, 2), selfterm_maxm = 1)
        plane, axis = heading_bases(maxdeg = (2,), maxm = 1, maxa = 1, history = hz)
        m3 = FrameModel(plane, ones(length(plane)), axis, zeros(length(axis)))
        e3(pv) = Environment3(SVector(0.0, 0.0, 1.0), SVector{3, Float64}[], SVector{3, Float64}[]; past_v = [pv], dt = 0.1)
        straight = e3(SVector(0.0, 0.0, 1.0))
        p = predict(m3, straight)
        Qz = SMatrix{3, 3}(cos(0.7), sin(0.7), 0.0, -sin(0.7), cos(0.7), 0.0, 0.0, 0.0, 1.0)
        @test predict(m3, rotate(straight, Qz)) ≈ Qz * p atol = 1.0e-12
        @test norm(p[1:2]) < 1.0e-12
        @test all(norm(predict(m3, e3(SVector(1.0e-7cos(θ), 1.0e-7sin(θ), 1.0))) - p) < 1.0e-5 for θ in (0.0, 1.0, 2.5))
        b2 = dACEBasis(maxdeg = (2,), history = hz)
        m2 = dACEModel(b2, ones(length(b2)))
        e2(pv) = Environment(SVector(0.8, 0.3), SVector{2, Float64}[], SVector{2, Float64}[]; past_v = [pv], dt = 0.1)
        @test norm(predict(m2, e2(SVector(1.0e-7, 0.0))) - predict(m2, e2(SVector(0.0, 1.0e-7)))) < 1.0e-5

        # time steps and scales must be positive and finite
        X, V = randn(Xoshiro(3), 2, 6, 4), randn(Xoshiro(4), 2, 6, 4)
        @test_throws ArgumentError trajectory_data(X, V, 0.0; cutoff = 2.0)
        ha = History(selfterm = 1, use = :acceleration, selfterm_degree = (2, 0))
        ba = dACEBasis(maxdeg = (2,), history = ha)
        v = SVector(0.8, 0.3)
        @test_throws ArgumentError evaluate(ba, Environment(v, SVector{2, Float64}[], SVector{2, Float64}[]; past_v = [v], dt = 0.0))
        @test_throws ArgumentError rollout(mp, X[:, :, 1], V[:, :, 1], 0.0, 2)
        # a rollout whose velocities overflow stops with a RolloutDivergence
        @test_throws RolloutDivergence rollout(dACEModel(mp.basis, fill(1.0e308, length(mp.c))), X[:, :, 1], V[:, :, 1], 0.1, 3)
        @test occursin("step 3", sprint(showerror, RolloutDivergence(3)))
        @test_throws ArgumentError dACEBasis(maxdeg = (2,), vscale = 0.0)
        @test_throws ArgumentError dACE.FrameBasis(maxdeg = (2,), cutoff = Inf)

        # rotation by a dense matrix, max_order of a basis without functions
        env3 = envs3[1]
        @test rotate(env3, Matrix(Qz)).v ≈ rotate(env3, Qz).v && rotate(env3, Matrix(Qz)).r ≈ rotate(env3, Qz).r
        @test dACE.max_order(dACEBasis(maxdeg = (2,), filter = (ψ, φs) -> false)) == 0

        # 3D prior = nothing, per-part λself in fit_path, parts that must agree
        @test fit(bs, envs3, accs3; λ = 1.0e-3, prior = nothing).c_plane ≈ fit(bs, envs3, accs3; λ = 1.0e-3, prior = (nothing, nothing)).c_plane
        mpath = fit_path(bs, envs3, accs3; λs = [1.0e-3], λself = (plane = 1.0e-3, axis = 1.0e-4))[1]
        # both parts are joint (no self term): the λself they used is λ
        @test mpath.λself == (plane = 1.0e-3, axis = 1.0e-3)
        @test only(fit_path(bp, envs, accs; λs = [1.0e-2], λself = 0.5)).λself == 1.0e-2
        @test only(fit_path(bh, henvs, haccs; λs = [1.0e-3], λself = 1.0e-4, reflections = true)).λself == 1.0e-4
        other = heading_bases(maxdeg = (4, 6), maxl = 0, maxm = 1, maxa = 1, maxb = 1, cutoff = 2interaction_radius(threebody))
        @test_throws ArgumentError design_statistics((bs[1], other[2]), envs3, accs3)
        @test_throws DimensionMismatch FrameModel(bs[1], ones(3), bs[2], ones(length(bs[2])))

        # CompositeBasis contributions (self model in the 1-body term) and rollout
        ib = dACEBasis(maxdeg = (2, 4), maxl = 0, maxm = 1, cutoff = interaction_radius(pair))
        cb = CompositeBasis(ib, dACEBasis(maxdeg = (2,), history = h))
        mc = fit(cb, henvs, haccs; λ = 1.0e-6, reflections = true)
        contrib = body_order_contributions(mc, henvs[1])
        @test length(contrib) == 2 && sum(contrib) ≈ predict(mc, henvs[1])
        @test dACE.history_of(mc) == h
        pos, vel = rollout(mc, sims[1].pos[:, :, 3], sims[1].vel[:, :, 3], sims[1].dt, 2; extent = sims[1].extent, periodic = true)
        @test size(pos, 3) == 3 && all(isfinite, vel)

        # the persistence is 0, not NaN, when the lagged residual vanishes
        @test dACE.persistence(1.0, 0.0, 1.0) == 0.0 && dACE.persistence(0.5, 1.0, 1.0) == 0.5
    end

    @testset "regressions: frames, shapes, workspaces and history" begin
        envs3, accs3 = swarm_data3(threebody, 1:1)
        bs = heading_bases(maxdeg = (4, 6), maxl = 0, maxm = 1, maxa = 1, maxb = 1, cutoff = interaction_radius(threebody))
        # the plane part has charge 1, the axis part charge 0
        @test_throws ArgumentError FrameModel(bs[2], zeros(length(bs[2])), bs[1], zeros(length(bs[1])))
        @test_throws ArgumentError design_statistics((bs[2], bs[1]), envs3, accs3)
        @test FrameModel(bs[1], zeros(length(bs[1])), bs[2], zeros(length(bs[2]))) isa FrameModel
        # snapshots and trajectories of one shape, past snapshots like the current one, frames in range
        @test_throws DimensionMismatch environments(zeros(2, 1), zeros(2, 2); cutoff = 2.0)
        @test_throws DimensionMismatch environments(zeros(3, 2), zeros(2, 2); cutoff = 2.0)
        @test_throws ArgumentError environments(zeros(4, 2), zeros(4, 2); cutoff = 2.0)
        hv = History(selfterm = 1)
        X, V = randn(Xoshiro(2), 2, 4), randn(Xoshiro(3), 2, 4)
        @test_throws DimensionMismatch environments(X, V; cutoff = 2.0, history = hv, past_velocities = [zeros(2, 3)], dt = 0.1)
        @test length(environments(X, V; cutoff = 2.0, history = hv, past_velocities = [zeros(2, 4)], dt = 0.1)) == 4
        P, W = randn(Xoshiro(4), 2, 4, 5), randn(Xoshiro(5), 2, 4, 5)
        @test_throws DimensionMismatch trajectory_data(P, W[:, 1:3, :], 0.1; cutoff = 2.0)
        @test_throws ArgumentError trajectory_data(P, W, 0.1; cutoff = 2.0, times = 1:5)
        @test length(first(trajectory_data(P, W, 0.1; cutoff = 2.0))) == 4 * 4
        # an agent at rest faces its latest non-zero past velocity (equivariant); one that has not moved has no
        # charged self features
        hz = History(selfterm = 1:2, selfterm_degree = (2, 2), selfterm_maxm = 1)
        b1 = dACEBasis(maxdeg = (2,), history = hz)
        m1 = dACEModel(b1, ones(length(b1)))
        o = SVector(0.0, 0.0)
        still(pv) = Environment(o, SVector{2, Float64}[], SVector{2, Float64}[]; past_v = pv, dt = 0.1)
        @test predict(m1, still([o, o])) == o
        for pv in ([SVector(0.6, -0.2), SVector(0.5, 0.1)], [o, SVector(-0.3, 0.4)])
            e = still(pv)
            @test predict(m1, rotate(e, 0.7)) ≈ rotation(0.7) * predict(m1, e)
            @test norm(predict(m1, e)) > 0
        end
        # 1 in 3D: the heading frame of an agent at rest is that of its latest non-zero past velocity; without one,
        # every heading-frame feature is 0
        hp3, ha3 = heading_bases(maxdeg = (2, 2), maxa = 1, history = hz)
        m3 = FrameModel(hp3, ones(length(hp3)), ha3, ones(length(ha3)))
        o3 = SVector(0.0, 0.0, 0.0)
        still3(pv) = Environment3(o3, [SVector(1.0, 0.5, -0.3)], [SVector(0.1, 0.0, 0.2)]; past_v = pv, dt = 0.1)
        @test predict(m3, still3([o3, o3])) == o3 && all(iszero, evaluate(hp3, still3([o3, o3])))
        Qr = SMatrix{3, 3}(Matrix(qr(randn(Xoshiro(8), 3, 3)).Q))
        Qr = det(Qr) < 0 ? -Qr : Qr
        for pv in ([SVector(0.2, 0.6, -0.1), SVector(0.1, 0.5, 0.0)], [o3, SVector(-0.3, 0.4, 0.2)])
            e = still3(pv)
            @test predict(m3, rotate(e, Qr)) ≈ Qr * predict(m3, e)
            @test norm(predict(m3, e)) > 0
        end
        # has_history follows the basis-history protocol, also for composites
        h = History(selfterm = 1:2, selfterm_degree = (4, 2), selfterm_maxm = 1)
        ib = dACEBasis(maxdeg = (2, 4), maxl = 0, maxm = 1)
        @test dACE.has_history(dACEModel(CompositeBasis(ib, dACEBasis(maxdeg = (2,), history = h)), zeros(length(ib) + length(dACEBasis(maxdeg = (2,), history = h)))))
        @test !dACE.has_history(dACEModel(CompositeBasis(ib, dACEBasis(maxdeg = (2,))), zeros(length(ib) + length(dACEBasis(maxdeg = (2,))))))
    end
end
