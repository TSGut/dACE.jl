using dACESwarms, Test, LinearAlgebra, StaticArrays

# brute-force neighbours
function neighbours(tr, t, i)
    D = size(tr.pos, 1)
    L = SVector{D}(tr.extent)
    xi = SVector{D}(tr.pos[:, i, t])
    out = Tuple{SVector{D, Float64}, SVector{D, Float64}}[]
    for j in axes(tr.pos, 2)
        j == i && continue
        dx = SVector{D}(tr.pos[:, j, t]) - xi
        tr.periodic && (dx = dx - L .* round.(dx ./ L))
        norm(dx) <= interaction_radius(tr.rule) && push!(out, (dx, SVector{D}(tr.vel[:, j, t])))
    end
    return out
end

@testset "dACESwarms" begin
    for rule in (Boids(), DOrsogna(), DOrsogna(κ = 2.0), DOrsogna(κv = 2.0), DOrsogna(κv = 2.0, κ4 = 4.0), DOrsogna(averaged = true), Zonal(), Boids(smoothing = 0.5), Zonal(smoothing = 0.5),
                 DOrsogna(chirality = 0.3), DOrsogna(ω = 0.5), DOrsogna(averaged = true, lateral = 0.5))
        @testset "$(nameof(typeof(rule))): data are an exact sample of the rule" begin
            tr = simulate(rule; n_agents = 40, extent = (30.0, 30.0), nsteps = 50, seed = 1, speeds = (0.5, 2.0))
            @test size(tr) == (2, 40, 51)
            L = collect(tr.extent)
            for t in 1:50, i in 1:40
                v = velocity_update(tr.rule, SVector{2}(tr.vel[:, i, t]), neighbours(tr, t, i), tr.dt)
                @test v ≈ tr.vel[:, i, t + 1] atol = 1.0e-10
                @test mod.(tr.pos[:, i, t] + tr.dt * v, L) ≈ tr.pos[:, i, t + 1] atol = 1.0e-10
            end
        end
    end

    for rule in (DOrsogna(), DOrsogna(κv = 2.0, κ4 = 4.0), Boids(), Zonal())
        @testset "3D $(nameof(typeof(rule))): data are an exact sample of the rule" begin
            tr = simulate(rule; n_agents = 30, extent = (30.0, 30.0, 30.0), nsteps = 30, seed = 2, spread = 10.0, speeds = (0.5, 2.0), dt = 0.05)
            @test size(tr) == (3, 30, 31)
            L = collect(tr.extent)
            for t in 1:30, i in 1:30
                v = velocity_update(tr.rule, SVector{3}(tr.vel[:, i, t]), neighbours(tr, t, i), tr.dt)
                @test v ≈ tr.vel[:, i, t + 1] atol = 1.0e-10
                @test mod.(tr.pos[:, i, t] + tr.dt * v, L) ≈ tr.pos[:, i, t + 1] atol = 1.0e-10
            end
        end
    end

    @testset "mirror image of a chiral rule is the rule of opposite chirality" begin
        σ(x) = SVector(x[1], -x[2])
        v = SVector(0.9, 0.4)
        nb = [(SVector(1.2, 0.7), SVector(0.3, 1.1)), (SVector(-0.8, 2.1), SVector(1.0, -0.2)), (SVector(2.5, -1.4), SVector(-0.4, 0.6))]
        mirrored = [(σ(dx), σ(vj)) for (dx, vj) in nb]
        for (θ, ω, b) in ((0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5), (0.3, 0.5, 0.5))
            rule(s) = DOrsogna(averaged = true, chirality = s * θ, ω = s * ω, lateral = s * b)
            @test acceleration(rule(1), σ(v), mirrored) ≈ σ(acceleration(rule(-1), v, nb))
            (θ, ω, b) == (0.0, 0.0, 0.0) || @test !(acceleration(rule(1), σ(v), mirrored) ≈ σ(acceleration(rule(1), v, nb)))
        end
    end

    @testset "3D headings lie in the prescribed cap" begin
        tr = simulate(DOrsogna(); n_agents = 200, extent = (100.0, 100.0, 100.0), nsteps = 0, seed = 3, heading_spread = π / 2)
        h = [normalize(SVector{3}(tr.vel[:, i, 1])) for i in 1:200]
        n0 = normalize(sum(h))
        @test minimum(dot(x, n0) for x in h) > cos(π / 4) - 0.05
    end

    @testset "prescribed initial conditions" begin
        pos0, vel0 = [10.0 12.0; 10.0 10.5], [1.0 -1.0; 0.0 0.2]
        tr = simulate(DOrsogna(); init = (pos0, vel0), extent = (30.0, 30.0), nsteps = 3, dt = 0.05)
        @test tr.pos[:, :, 1] == pos0 && tr.vel[:, :, 1] == vel0 && size(tr) == (2, 2, 4)
    end
end
