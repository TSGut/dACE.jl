# recovery of simulated D'Orsogna swarms by body order, 2D and 3D

@testset "swarms" begin
    @testset "pairwise D'Orsogna swarm" begin
        envs, accs = swarm_data(pair, 1:8)
        test_envs, test_accs = swarm_data(pair, 101:103)
        m2 = fit(dACEBasis(maxdeg = (4, 8)), envs, accs)
        @test rmse(m2, test_envs, test_accs; relative = true) < 0.01
        # the true rule as a model of one's own, through the generic prediction path
        @test rmse(TrueModel(pair), test_envs, test_accs; relative = true) < 1.0e-6

        m3 = fit(dACEBasis(maxdeg = (4, 8, 4)), envs, accs)
        @test rmse(m3, test_envs, test_accs; relative = true) < 0.01
        c = [body_order_contributions(m3, e) for e in test_envs]
        @test sqrt(mean(norm(ci[3])^2 for ci in c)) < 0.01 * sqrt(mean(norm.(test_accs) .^ 2))

        tr = simulate(
            pair; n_agents = 30, nsteps = 100, seed = 999, extent = (100.0, 100.0),
            spread = 12.0, speeds = (0.2, 2.5), dt = 0.05,
        )
        pos, _ = rollout(m2, tr.pos[:, :, 1], tr.vel[:, :, 1], tr.dt, 100; extent = tr.extent, periodic = true)
        @test maximum(abs, pos - tr.pos) < 0.1
    end

    @testset "3-body D'Orsogna swarm" begin
        envs, accs = swarm_data(threebody, 1:12)
        test_envs, test_accs = swarm_data(threebody, 101:103)
        m3 = fit(dACEBasis(maxdeg = (4, 6, 6)), envs, accs)
        @test rmse(m3, test_envs, test_accs; relative = true) < 0.03

        learned = [body_order_contributions(m3, e) for e in test_envs]
        truth = true_orders.(Ref(threebody), test_envs)
        for N in 1:3
            @test relrms(getindex.(learned, N), getindex.(truth, N)) < 0.06
        end
    end

    @testset "4-body D'Orsogna swarm (no 3-body term)" begin
        envs, accs = swarm_data(fourbody, 1:24)
        test_envs, test_accs = swarm_data(fourbody, 101:103)
        m4 = fit(small((4, 8, 6, 6)), envs, accs)
        @test rmse(m4, test_envs, test_accs; relative = true) < 0.015

        learned = [body_order_contributions(m4, e) for e in test_envs]
        truth = true_orders.(Ref(fourbody), test_envs)
        # absent 3-body order
        @test rms(getindex.(learned, 3)) < 0.06 * rms(getindex.(learned, 4))
        for N in (1, 2, 4)
            @test relrms(getindex.(learned, N), getindex.(truth, N)) < 0.05
        end
    end

    @testset "mixed 2-, 3- and 4-body D'Orsogna swarm" begin
        envs, accs = swarm_data(mixed, 1:32)
        test_envs, test_accs = swarm_data(mixed, 101:103)
        m4 = fit(small((4, 8, 6, 6)), envs, accs)
        @test rmse(m4, test_envs, test_accs; relative = true) < 0.015

        learned = [body_order_contributions(m4, e) for e in test_envs]
        truth = true_orders.(Ref(mixed), test_envs)
        @test all(N -> rms(getindex.(truth, N)) > 0.2 * rms(test_accs), 1:4)
        for N in 1:4
            @test relrms(getindex.(learned, N), getindex.(truth, N)) < 0.12
        end
    end

    @testset "3D heading frame" begin
        rng3 = Xoshiro(2)
        randrot() = (Q = Matrix(qr(randn(rng3, 3, 3)).Q); det(Q) < 0 && (Q[:, 1] .*= -1); SMatrix{3, 3}(Q))
        env3 = Environment3(
            SVector(0.3, -1.1, 0.7),
            SVector{3, Float64}[(0.4 + 4rand(rng3)) * normalize(randn(rng3, SVector{3})) for _ in 1:5],
            SVector{3, Float64}[randn(rng3, SVector{3}) for _ in 1:5],
        )
        plane, axis = heading_bases(maxdeg = (2, 3, 3, 2), maxl = 1, maxa = 2, maxb = 1)
        Q = randrot()
        for b in (plane, axis)
            @test evaluate(b, env3) ≈ brute_force(b, env3)
            @test dACE.evaluate_direct(b, env3) ≈ evaluate(b, env3)
        end
        @test evaluate(axis, rotate(env3, Q)) ≈ evaluate(axis, env3)
        ratio = evaluate(plane, rotate(env3, Q)) ./ evaluate(plane, env3)
        @test all(r -> r ≈ ratio[1], ratio) && abs(ratio[1]) ≈ 1
        m = FrameModel(plane, randn(rng3, ComplexF64, length(plane)), axis, randn(rng3, ComplexF64, length(axis)))
        @test predict(m, rotate(env3, Q)) ≈ Q * predict(m, env3)
        @test sum(body_order_contributions(m, env3)) ≈ predict(m, env3)
        @test_throws DomainError dACE.heading_frame(SVector(0.0, 0.0, 0.0))
    end

    @testset "3D vertical frame" begin
        rng3 = Xoshiro(3)
        env3 = Environment3(
            SVector(0.3, -1.1, 0.4),
            SVector{3, Float64}[(0.4 + 4rand(rng3)) * normalize(randn(rng3, SVector{3})) for _ in 1:5],
            SVector{3, Float64}[0.5randn(rng3, SVector{3}) for _ in 1:5],
        )
        hor, ver = vertical_bases(maxdeg = (2, 3, 3, 2), maxl = 1, maxa = 2, maxb = 1)
        for b in (hor, ver)
            @test evaluate(b, env3) ≈ brute_force(b, env3)
            @test dACE.evaluate_direct(b, env3) ≈ evaluate(b, env3)
        end
        α = 0.7
        Rz = SMatrix{3, 3}(cos(α), sin(α), 0, -sin(α), cos(α), 0, 0, 0, 1)
        Rx = SMatrix{3, 3}(1, 0, 0, 0, cos(α), sin(α), 0, -sin(α), cos(α))
        @test evaluate(ver, rotate(env3, Rz)) ≈ evaluate(ver, env3)
        @test evaluate(hor, rotate(env3, Rz)) ≈ cis(α) .* evaluate(hor, env3)
        m = FrameModel(hor, randn(rng3, ComplexF64, length(hor)), ver, randn(rng3, ComplexF64, length(ver)))
        @test predict(m, rotate(env3, Rz)) ≈ Rz * predict(m, env3)
        @test !(predict(m, rotate(env3, Rx)) ≈ Rx * predict(m, env3))
        @test sum(body_order_contributions(m, env3)) ≈ predict(m, env3)
        @test predict(m, Environment3(zero(env3.v), env3.r, env3.u)) isa SVector{3}
        γ = smoothness_prior.((hor, ver))
        @test length.(γ) == (length(hor), length(ver)) && all(g -> all(>=(1), g), γ)
    end

    @testset "3D mixed 2-, 3- and 4-body D'Orsogna swarm" begin
        envs, accs = swarm_data3(mixed, 1:16)
        test_envs, test_accs = swarm_data3(mixed, 101:103)
        m4 = fit(heading_bases(maxdeg = (4, 8, 5, 4), maxa = 1), envs, accs)
        @test rmse(m4, test_envs, test_accs; relative = true) < 0.025
        tr = simulate(mixed; n_agents = 20, nsteps = 5, seed = 7, extent = (100.0, 100.0, 100.0), spread = 10.0, speeds = (0.2, 2.5), dt = 0.05)
        pos, vel = rollout(m4, tr.pos[:, :, 1], tr.vel[:, :, 1], tr.dt, 5; extent = tr.extent, periodic = true)
        @test size(pos) == size(tr.pos) && all(isfinite, pos) && all(isfinite, vel)
        learned = [body_order_contributions(m4, e) for e in test_envs]
        truth = true_orders.(Ref(mixed), test_envs)
        for N in 1:4
            @test relrms(getindex.(learned, N), getindex.(truth, N)) < 0.15
        end
    end
end
