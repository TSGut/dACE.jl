# fitting: streamed QR, normal equations, paths, AR(1), the evidence, statistics

@testset "fitting" begin
    @testset "streamed QR and normal equations agree with a direct QR solve" begin
        envs, accs = swarm_data(pair, 1:2)
        b = dACEBasis(maxdeg = (4, 6))
        m_direct = fit(b, envs, accs; solver = dACE.ACEfit.QR(lambda = 1.0e-8))
        for method in (:qr, :normal_equations)
            @test relrms(predict(fit(b, envs, accs; λ = 1.0e-6, method), envs), predict(m_direct, envs)) < 1.0e-6
        end
        @test_throws ArgumentError fit(b, envs, accs; method = :cholesky)
    end

    @testset "regularisation path" begin
        envs, accs = swarm_data(pair, 1:2)
        b = dACEBasis(maxdeg = (4, 6))
        for method in (:qr, :normal_equations), reflections in (false, true)
            ms = fit_path(b, envs, accs; λs = (1.0e-6, 1.0e-2), method, reflections)
            @test ms[1].c ≈ fit(b, envs, accs; λ = 1.0e-6, method, reflections).c
            @test ms[2].c ≈ fit(b, envs, accs; λ = 1.0e-2, method, reflections).c
        end
    end

    @testset "streamed QR least squares" begin
        rngq = Xoshiro(11)
        # the row-block updates give the triangular factor of the whole design, real and complex
        for T in (Float64, ComplexF64)
            W = randn(rngq, T, 523, 8)
            S = dACE.qr_statistics(W, 7; chunksize = 50)
            @test S.R' * S.R ≈ W' * W
            @test S.R == UpperTriangular(S.R) && S.n == 523
        end
        # agreement with the normal equations on a well-conditioned design, including a vanishing
        # column, chunking, pooling and the λ path
        B = randn(rngq, ComplexF64, 400, 9)
        B[:, 4] .= 0
        t = randn(rngq, ComplexF64, 400)
        S = dACE.qr_statistics([B t], 9; chunksize = 37)
        pooled = dACE.qr_statistics([B[1:150, :] t[1:150]], 9) + dACE.qr_statistics([B[151:end, :] t[151:end]], 9)
        for reflections in (false, true), prior in (nothing, collect(1.0:9.0)), λ in (1.0e-6, 0.1)
            c = dACE.lsq_solve(S; λ, prior, reflections)
            @test c ≈ dACE.solve_normal_equations(B' * B, B' * t; λ, prior, reflections) rtol = 1.0e-9
            @test c[4] == 0 && (!reflections || all(iszero, imag(c)))
            @test dACE.lsq_solve(pooled; λ, prior, reflections) ≈ c rtol = 1.0e-9
            @test dACE.lsq_path(S; λs = (0.5, λ), prior, reflections)[2] ≈ c rtol = 1.0e-9
            @test dACE.lsq_sse(S, c) ≈ sum(abs2, B * c - t) rtol = 1.0e-10
        end
        # an ill-conditioned design (normalised singular values 1 … 1e-7) with a tiny λ: the QR solve keeps
        # full accuracy
        K, N = 30, 2000
        U = Matrix(qr(randn(rngq, N, K)).Q)[:, 1:K]
        V = Matrix(qr(randn(rngq, K, K)).Q)
        Bi = U * Diagonal(10.0 .^ range(0, -7, length = K)) * V'
        ti = Bi * randn(rngq, K) + 1.0e-10 * randn(rngq, N)
        λ = 1.0e-12
        reference = setprecision(512) do
            Bb = big.(Bi)
            s = [norm(Bb[:, k]) for k in 1:K]
            A = Bb ./ s'
            Float64.(((A' * A + λ^2 * I) \ (A' * big.(ti))) ./ s)
        end
        cq = real(dACE.lsq_solve(dACE.qr_statistics(complex.([Bi ti]), K; chunksize = 300); λ, reflections = true))
        @test norm(cq - reference) / norm(reference) < 1.0e-8
        @test real(dACE.lsq_solve(dACE.qr_statistics([Bi ti], K; chunksize = 300); λ, reflections = true)) ≈ cq
        # model level: both methods, both symmetries, with the smoothness prior
        envs, accs = swarm_data(pair, 1:2)
        b = dACEBasis(maxdeg = (4, 6))
        γ = smoothness_prior(b)
        for reflections in (false, true)
            mq = fit(b, envs, accs; λ = 1.0e-4, prior = γ, reflections)
            mn = fit(b, envs, accs; λ = 1.0e-4, prior = γ, reflections, method = :normal_equations)
            @test mq.c ≈ mn.c rtol = 1.0e-6
        end
        St = dACE.qr_statistics(b, envs, [complex(a[1], a[2]) for a in accs]; chunksize = 333)
        m = fit(b, envs, accs; λ = 1.0e-4)
        @test dACE.lsq_sse(St, m.c) ≈ sum(norm(predict(m, e) - a)^2 for (e, a) in zip(envs, accs)) rtol = 1.0e-8
        @test_throws ArgumentError dACE.ar_solve(St)
        # rows added from an external design, column subsets and the target norm
        Se = dACE.QRStatistics(ComplexF64, 9)
        dACE.add_rows!(dACE.add_rows!(Se, [B[1:200, :] t[1:200]]), [B[201:end, :] t[201:end]])
        @test Se.n == 400 && Se.R' * Se.R ≈ [B t]' * [B t]
        @test dACE.target_norm2(Se) ≈ sum(abs2, t)
        cols = [1, 3, 5, 6, 9]
        Ss = dACE.select_columns(Se, cols)
        @test Ss.K == 5 && Ss.R == UpperTriangular(Ss.R)
        @test dACE.lsq_solve(Ss; λ = 1.0e-3) ≈ dACE.solve_normal_equations(B[:, cols]' * B[:, cols], B[:, cols]' * t; λ = 1.0e-3) rtol = 1.0e-9
        @test_throws DimensionMismatch dACE.add_rows!(Se, B)
        @test_throws ArgumentError dACE.add_rows!(dACE.QRStatistics(Float64, 9), [B t])
    end

    @testset "reflection-symmetric fit" begin
        envs, accs = swarm_data(threebody, 1:12)
        test_envs, test_accs = swarm_data(threebody, 101:103)
        m = fit(dACEBasis(maxdeg = (4, 8, 6), maxl = 0, maxm = 1), envs, accs; reflections = true)
        @test all(iszero ∘ imag, m.c)
        @test rmse(m, test_envs, test_accs; relative = true) < 0.02
    end

    @testset "AR(1) noise model" begin
        rngn = Xoshiro(6)
        envs, F, penvs, pF = Environment{Float64}[], SVector{2, Float64}[], Environment{Float64}[], SVector{2, Float64}[]
        for seed in 1:6
            tr = simulate(pair; n_agents = 30, nsteps = 60, seed, extent = (100.0, 100.0), spread = 12.0, speeds = (0.2, 2.5), dt = 0.05)
            d = trajectory_data(tr.pos, tr.vel, tr.dt; cutoff = interaction_radius(pair), extent = tr.extent, periodic = true, times = 2:2:60, lagged = true)
            append!(envs, d[1]); append!(F, d[2]); append!(penvs, d[3]); append!(pF, d[4])
        end
        @test length(envs) == length(penvs) == length(F) == length(pF) == 6 * 30 * 30
        # synthetic targets: force + persistent noise η(t) = 0.8 η(t-1) + ξ (independent of the state)
        ηp = [0.5randn(rngn, SVector{2}) for _ in F]
        η = [0.8x + 0.1randn(rngn, SVector{2}) for x in ηp]
        b = dACEBasis(maxdeg = (4, 6))
        m = fit_ar1(b, envs, F .+ η, penvs, pF .+ ηp; λ = 1.0e-2, prior = nothing)
        @test m isa ARModel && abs(m.ρ - 0.8) < 0.03
        # the AR model predicts and rolls out as its force law: the noise model is not part of the force
        @test predict(m, envs[1]) == predict(m.model, envs[1]) && predict(m, envs[1:3]) == predict(m.model, envs[1:3])
        tr = simulate(pair; n_agents = 30, nsteps = 5, seed = 1, extent = (100.0, 100.0), spread = 12.0, speeds = (0.2, 2.5), dt = 0.05)
        @test rollout(m, tr.pos[:, :, 1], tr.vel[:, :, 1], tr.dt, 5; extent = tr.extent, periodic = true) ==
            rollout(m.model, tr.pos[:, :, 1], tr.vel[:, :, 1], tr.dt, 5; extent = tr.extent, periodic = true)
        # λ by the evidence of the quasi-differenced fit (default)
        me = fit_ar1(b, envs, F .+ η, penvs, pF .+ ηp; prior = nothing)
        @test me.model.λ in dACE.EVIDENCE_GRID && abs(me.ρ - 0.8) < 0.03
        @test rmse(m, envs, F; relative = true) < 0.25
        # one-step prediction needs the past
        @test sqrt(sum(norm(predict(m, e, pe, pa) - a)^2 for (e, pe, pa, a) in zip(envs, penvs, pF .+ ηp, F .+ η)) / length(envs)) <
            rmse(m.model, envs, F .+ η)
        S = dACE.ar_statistics(b, envs, penvs, F .+ η, pF .+ ηp)
        direct = sum(norm(predict(m, e, pe, pa) - a)^2 for (e, pe, pa, a) in zip(envs, penvs, pF .+ ηp, F .+ η))
        @test dACE.ar_sse(S, m.model.c, m.ρ) ≈ direct rtol = 1.0e-6
        # streamed QR (default) and normal equations give the same AR(1) fit and one-step errors
        @test S isa dACE.QRStatistics && S.lagged
        Sn = dACE.ar_statistics(b, envs, penvs, F .+ η, pF .+ ηp; method = :normal_equations)
        @test dACE.ar_sse(Sn, m.model.c, m.ρ) ≈ direct rtol = 1.0e-6
        # iters = 0: the fit at ρ = 0 (no noise model), by either method
        @test dACE.ar_solve(S; λ = 1.0e-2, iters = 0)[1] ≈ dACE.ar_solve(Sn; λ = 1.0e-2, iters = 0)[1] rtol = 1.0e-6
        @test dACE.ar_solve(S; λ = 1.0e-2, iters = 0)[2] == 0
        cx(a) = complex(a[1], a[2])
        Sq = dACE.qr_statistics(b, envs, cx.(F .+ η); past = (penvs, cx.(pF .+ ηp)), chunksize = 500)
        @test dACE.ar_sse(Sq, m.model.c, m.ρ) ≈ direct rtol = 1.0e-8
        @test dACE.target_norm2(Sq) ≈ sum(norm.(F .+ η) .^ 2)
        Sq2 = dACE.select_columns(Sq, 1:3)
        @test dACE.ar_solve(Sq2; λ = 1.0e-2)[1] ≈ dACE.ar_solve(dACE.select_columns(S, 1:3); λ = 1.0e-2)[1] rtol = 1.0e-8
        for reflections in (false, true)
            mq = fit_ar1(b, envs, F .+ η, penvs, pF .+ ηp; λ = 1.0e-2, reflections)
            mn = fit_ar1(b, envs, F .+ η, penvs, pF .+ ηp; λ = 1.0e-2, reflections, method = :normal_equations)
            @test mq.ρ ≈ mn.ρ rtol = 1.0e-6
            @test mq.model.c ≈ mn.model.c rtol = 1.0e-6
        end
        @test_throws ArgumentError dACE.lsq_solve(Sq)
    end

    @testset "regularisation by the evidence" begin
        # the profiled log evidence K log λ² - Σ log(λ² + σ²) - n log Q of the SVD is that of the Gaussian model
        # b ~ N(0, β⁻¹ (I + Ã Ãᴴ / λ²)) with β at its optimum, and its maximiser is the chosen λ
        rng = Xoshiro(5)
        A = randn(rng, ComplexF64, 120, 8) .* (2.0 .^ -(0:7))'
        b = A * randn(rng, ComplexF64, 8) .+ 0.05 .* randn(rng, ComplexF64, 120)
        D = dACE.scaled_svd(A, b)
        Ã = A ./ D.w'
        direct(λ) = (C = I + Ã * Ã' / λ^2; -logdet(Hermitian(C)) - 120log(real(b' * (C \ b))))
        λs = 10.0 .^ (-4:0.01:1)
        @test dACE.evidence_lambda(D, b, 120; λs) == λs[argmax(direct.(λs))]
        # fit chooses it by default and records it, a number overrides it
        envs, accs = swarm_data(pair, 1:3)
        bp = dACEBasis(maxdeg = (4, 6), maxl = 0, maxm = 1, cutoff = interaction_radius(pair))
        m = fit(bp, envs, accs)
        @test m.λ in dACE.EVIDENCE_GRID && m.λself == m.λ && fit(bp, envs, accs; λ = m.λ).c ≈ m.c
        @test fit(bp, envs, accs; λ = 1.0e-3).λ == 1.0e-3
        @test fit(bp, design_statistics(bp, envs, accs)).c ≈ m.c
        # one λ for several data sets: for two copies of one data set, the λ of that data set
        Sp = design_statistics(bp, envs, accs)
        @test dACE.shared_evidence_lambda([Sp, Sp]; prior = smoothness_prior(bp)) == m.λ
        @test_throws ArgumentError fit(bp, envs, accs; method = :normal_equations)
        envs3, accs3 = swarm_data3(threebody, 1:2)
        bs = heading_bases(maxdeg = (4, 6), maxl = 0, maxm = 1, maxa = 1, maxb = 1, cutoff = interaction_radius(threebody))
        m3 = fit(bs, envs3, accs3)
        @test m3.λ.plane in dACE.EVIDENCE_GRID && m3.λ.axis in dACE.EVIDENCE_GRID
        @test fit(bs, envs3, accs3; λ = m3.λ).c_axis ≈ m3.c_axis
    end

    @testset "streamed fitting: design_statistics, select_lambda, fit from statistics" begin
        envs, accs = swarm_data(pair, 1:6)
        b = dACEBasis(maxdeg = (4, 6), maxl = 0, maxm = 1, cutoff = interaction_radius(pair))
        parts = [1:1800, 1801:3600, 3601:length(envs)]
        S = [design_statistics(b, envs[p], accs[p]) for p in parts]
        λs = 10.0 .^ (-6:1:0)
        sel = select_lambda(S[1] + S[2], S[3], b; λs, reflections = true)
        @test sel.λ in λs && sel.λself == sel.λ
        # the fit from pooled statistics is the fit on the pooled data
        @test fit(b, S[1] + S[2] + S[3]; sel..., reflections = true).c ≈ fit(b, envs, accs; λ = sel.λ, reflections = true).c
        # the selected λ minimises the validation error over the path
        val(λ) = rmse(fit(b, S[1] + S[2]; λ, reflections = true), envs[parts[3]], accs[parts[3]])
        @test val(sel.λ) <= minimum(val, λs) * (1 + 1.0e-9)
        # degree selection on validation data: the candidate with the lowest validation error, λ chosen per candidate
        cands = [dACEBasis(maxdeg = d, maxl = 0, maxm = 1, cutoff = interaction_radius(pair)) for d in ((4, 6), (6, 10))]
        tr_, va_ = 1:3600, 3601:length(envs)
        sb = select_basis(cands, envs[tr_], accs[tr_], envs[va_], accs[va_]; λs, reflections = true)
        manual = map(cands) do c
            sl = select_lambda(design_statistics(c, envs[tr_], accs[tr_]), design_statistics(c, envs[va_], accs[va_]), c; λs, reflections = true)
            rmse(fit(c, envs[tr_], accs[tr_]; λ = sl.λ, reflections = true), envs[va_], accs[va_]; relative = true)
        end
        @test sb.errors ≈ manual && sb.index == argmin(manual) && sb.basis === cands[sb.index]
        @test select_basis(cands[1:1], envs[tr_], accs[tr_], envs[va_], accs[va_]; λs).index == 1
        @test fit(sb.basis, sb.statistics; λ = sb.λ, reflections = true).c ≈ fit(sb.basis, envs, accs; λ = sb.λ, reflections = true).c
        # a degree ladder shares one design: the columns of a smaller degree are those of the larger, same values
        ladder = [dACEBasis(maxdeg = d, maxl = 0, maxm = 1, cutoff = interaction_radius(pair)) for d in ((4, 6, 4, 2), (4, 8, 6, 4))]
        cm = dACE.column_map(ladder...)
        @test !isnothing(cm) && length(cm) == length(ladder[1]) && allunique(cm)
        @test all(evaluate(ladder[2], e)[cm] ≈ evaluate(ladder[1], e) for e in envs[1:20])
        Sl = design_statistics(ladder[2], envs[tr_], accs[tr_])
        @test fit(ladder[1], dACE.restrict_statistics(Sl, cm); λ = 1.0e-4).c ≈ fit(ladder[1], envs[tr_], accs[tr_]; λ = 1.0e-4).c
        @test isnothing(dACE.column_map(ladder[1], dACEBasis(maxdeg = (4, 8, 6, 4), maxl = 0, maxm = 1, cutoff = 2interaction_radius(pair))))
        @test isnothing(dACE.column_map(ladder[2], ladder[1]))
        # with a self term: the staged selection
        h = History(selfterm = 1:2, selfterm_degree = (4, 2), selfterm_maxm = 1)
        tr = [simulate(pair; n_agents = 30, nsteps = 40, seed, extent = (100.0, 100.0), spread = 12.0, speeds = (0.2, 2.5), dt = 0.05) for seed in 1:3]
        d = [trajectory_data(t.pos, t.vel, t.dt; cutoff = interaction_radius(pair), extent = t.extent, periodic = true, times = 3:2:39, history = h) for t in tr]
        bh = dACEBasis(maxdeg = (4, 6), maxl = 0, maxm = 1, cutoff = interaction_radius(pair), history = h)
        Sh = [design_statistics(bh, x...) for x in d]
        hb = [dACEBasis(maxdeg = dg, maxl = 0, maxm = 1, cutoff = interaction_radius(pair), history = h) for dg in ((4, 6), (4, 8))]
        hm = dACE.column_map(hb...)
        @test !isnothing(hm) && all(evaluate(hb[2], e)[hm] ≈ evaluate(hb[1], e) for e in d[1][1][1:20])
        selh = select_lambda(Sh[1] + Sh[2], Sh[3], bh; λs, reflections = true)
        @test (selh.λ, selh.λself) == dACE.select_staged(Sh[1] + Sh[2], Sh[3], bh; λs, prior = smoothness_prior(bh), reflections = true)
        @test fit(bh, sum(Sh); selh..., reflections = true).c ≈
            fit(bh, reduce(vcat, first.(d)), reduce(vcat, last.(d)); λ = selh.λ, λself = selh.λself, reflections = true).c
        # 3D: statistics per part, λ per part, reflections
        envs3, accs3 = swarm_data3(threebody, 1:3)
        bs = heading_bases(maxdeg = (4, 6), maxl = 0, maxm = 1, maxa = 1, maxb = 1, cutoff = interaction_radius(threebody))
        n3 = length(envs3) ÷ 2
        S3a, S3b = design_statistics(bs, envs3[1:n3], accs3[1:n3]), design_statistics(bs, envs3[(n3 + 1):end], accs3[(n3 + 1):end])
        @test S3a isa dACE.FrameStatistics
        sel3 = select_lambda(S3a, S3b, bs; λs, reflections = true)
        @test sel3.λ.plane in λs && sel3.λ.axis in λs
        m3 = fit(bs, S3a + S3b; sel3..., reflections = true)
        @test m3.c_plane ≈ fit(bs, envs3, accs3; λ = sel3.λ.plane, reflections = true).c_plane
        @test m3.c_axis ≈ fit(bs, envs3, accs3; λ = sel3.λ.axis, reflections = true).c_axis
        @test all(iszero, imag.(m3.c_plane)) && all(iszero, imag.(m3.c_axis))
        ladder3 = [heading_bases(maxdeg = dg, maxl = 0, maxm = 1, maxa = 1, maxb = 1, cutoff = interaction_radius(threebody)) for dg in ((4, 6), (4, 8, 5))]
        sb3 = select_basis(ladder3, envs3[1:n3], accs3[1:n3], envs3[(n3 + 1):end], accs3[(n3 + 1):end]; λs, reflections = true)
        manual3 = map(ladder3) do c
            Sc = design_statistics(c, envs3[1:n3], accs3[1:n3])
            sl = select_lambda(Sc, design_statistics(c, envs3[(n3 + 1):end], accs3[(n3 + 1):end]), c; λs, reflections = true)
            rmse(fit(c, Sc; sl..., reflections = true), envs3[(n3 + 1):end], accs3[(n3 + 1):end]; relative = true)
        end
        @test !isnothing(dACE.column_map(ladder3...)) && sb3.errors ≈ manual3 && sb3.index == argmin(manual3)
        σ(x) = SVector(x[1], -x[2], x[3])
        @test all(predict(m3, reflect(e)) ≈ σ(predict(m3, e)) for e in envs3[1:20])
        # body orders
        b3 = dACEBasis(maxdeg = (2, 3, 3), maxl = 1, history = History(selfterm = 1))
        @test max_body_order(b3) == 3 && sort(unique(body_orders(b3))) == [1, 2, 3]
        @test count(==(1), body_orders(b3)) == count(f -> dACE.correlation_order(f) == 0, b3.funcs) + length(b3.selfterm)
        # environments: a history needs its past velocities and dt
        X, V = randn(Xoshiro(1), 2, 5), randn(Xoshiro(2), 2, 5)
        @test_throws ArgumentError environments(X, V; cutoff = 2.0, history = History(selfterm = 1:2), past_velocities = [V], dt = 0.1)
        @test_throws ArgumentError environments(X, V; cutoff = 2.0, history = History(selfterm = 1), past_velocities = [V])
        @test all(e -> isempty(e.past_v), environments(X, V; cutoff = 2.0, past_velocities = [V], dt = 0.1))
        @test all(e -> length(e.past_v) == 1 && isempty(e.past_vn), environments(X, V; cutoff = 2.0, history = History(selfterm = 1), past_velocities = [V, V], dt = 0.1))
    end

    @testset "3D smoothness prior in fit and fit_path" begin
        envs, accs = swarm_data3(threebody, 1:4)
        bs = heading_bases(maxdeg = (4, 6, 4), maxl = 0, maxm = 1, maxa = 1, maxb = 1)
        γ = smoothness_prior.(bs)
        ms = fit_path(bs, envs, accs; λs = (1.0e-4, 1.0e-2), prior = γ)
        @test ms[2].c_plane ≈ fit(bs, envs, accs; λ = 1.0e-2, prior = γ).c_plane
        @test !(ms[2].c_plane ≈ fit(bs, envs, accs; λ = 1.0e-2, prior = (nothing, nothing)).c_plane)
        @test ms[2].c_plane ≈ fit(bs, envs, accs; λ = 1.0e-2).c_plane   # the default prior
        mn = fit(bs, envs, accs; λ = 1.0e-2, prior = γ, method = :normal_equations)
        @test ms[2].c_plane ≈ mn.c_plane rtol = 1.0e-6
        @test ms[2].c_axis ≈ mn.c_axis rtol = 1.0e-6
        @test fit_path(bs, envs, accs; λs = (1.0e-2,), prior = γ, method = :normal_equations)[1].c_axis ≈ mn.c_axis
    end
end
