# shared fixtures of the test files: random environments, a brute-force evaluator, simulated data, rules

rotation(α) = SMatrix{2, 2}(cos(α), sin(α), -sin(α), cos(α))

function random_env(rng, J)
    r = [(4 * rand(rng) + 0.3) * SVector(cos(θ), sin(θ)) for θ in 2π * rand(rng, J)]
    u = [randn(rng, SVector{2}) for _ in 1:J]
    return Environment(randn(rng, SVector{2}), r, u)
end

# brute force over distinct tuples
function brute_force(basis, env)
    Φ = dACE.onep_values(basis, env)
    Ψ = dACE.centre_values(basis, env)
    J = length(env)
    return map(basis.funcs) do f
        N = dACE.correlation_order(f)
        s = zero(ComplexF64)
        for js in Iterators.product(ntuple(_ -> 1:J, N)...)
            allunique(js) || continue
            s += prod(Φ[js[t], f.ks[t]] for t in 1:N; init = one(ComplexF64))
        end
        Ψ[f.centre] * s
    end |> B -> isnothing(basis.selfterm) ? B :
        [B; dACE.self_values!(zeros(ComplexF64, length(basis.selfterm)), basis.selfterm, env, basis.charge, dACE.self_frame(basis, env))]
end

# an environment with a history of `H` past velocities of the agent and one of each neighbour
history_env(rng, e; H = 3, dt = 0.1) = Environment(e.v, e.r, e.u; past_v = [e.v + 0.5randn(rng, SVector{2}) for _ in 1:H],
    past_vn = [e.v + u + 0.5randn(rng, SVector{2}) for u in e.u], dt)
history_env3(rng, e; H = 3, dt = 0.1) = Environment3(e.v, e.r, e.u; past_v = [e.v + 0.5randn(rng, SVector{3}) for _ in 1:H],
    past_vn = [e.v + u + 0.5randn(rng, SVector{3}) for u in e.u], dt)

function swarm_data3(rule, seeds; n_agents = 40, nsteps = 60)
    envs, accs = Environment3{Float64}[], SVector{3, Float64}[]
    for seed in seeds
        tr = simulate(
            rule; n_agents, nsteps, seed, extent = (100.0, 100.0, 100.0), spread = 10.0,
            speeds = (0.2, 2.5), dt = 0.05,
        )
        e, a = trajectory_data(
            tr.pos, tr.vel, tr.dt; cutoff = interaction_radius(rule), extent = tr.extent,
            periodic = true, times = 1:2:nsteps,
        )
        append!(envs, e)
        append!(accs, a)
    end
    return envs, accs
end

function swarm_data(rule, seeds; n_agents = 30, nsteps = 60)
    envs, accs = Environment{Float64}[], SVector{2, Float64}[]
    for seed in seeds
        tr = simulate(
            rule; n_agents, nsteps, seed, extent = (100.0, 100.0), spread = 12.0,
            speeds = (0.2, 2.5), dt = 0.05,
        )
        e, a = trajectory_data(
            tr.pos, tr.vel, tr.dt; cutoff = interaction_radius(rule), extent = tr.extent,
            periodic = true, times = 1:2:nsteps,
        )
        append!(envs, e)
        append!(accs, a)
    end
    return envs, accs
end

relrms(x, y) = sqrt(sum(norm.(x - y) .^ 2) / sum(norm.(y) .^ 2))

# the benchmark rules and their true contributions of orders 0–3
pair, threebody = DOrsogna(), DOrsogna(κv = 2.0)
fourbody, mixed = DOrsogna(κ4 = 4.0), DOrsogna(κv = 2.0, κ4 = 4.0)
nbrs(e) = [(e.r[j], e.u[j] + e.v) for j in 1:length(e)]
function true_orders(rule, e)
    a(r) = acceleration(r, e.v, nbrs(e))
    a0, a1, a2 = a(DOrsogna(Cr = 0.0, Ca = 0.0, γ = 0.0)), a(pair), a(DOrsogna(κv = rule.κv))
    return [a0, a1 - a0, a2 - a1, a(rule) - a2]
end
rms(xs) = sqrt(mean(norm.(xs) .^ 2))

small(maxdeg) = dACEBasis(; maxdeg, maxl = 0, maxm = 1)
