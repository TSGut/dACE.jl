# julia -t 5 --project=scripts/paper scripts/synthetic/efficiency.jl
#
# Data efficiency (Figure 8): relative force error of the 4-body model on the clean test simulations of split 1
# against the number of training simulations (900 samples each), for the pairwise and 2+3+4-body rules, with
# observation noise ε = 0, 0.01, 0.1 (every recorded position and velocity shifted by a vector of random direction
# and length uniform in [0, ε]; targets by finite differences of the noisy velocities). λ by the evidence on all
# training samples, real coefficients.
# out: results/efficiency.jsonl (appended, one line per fit, so a partial run is kept)

include(joinpath(@__DIR__, "..", "paper", "setup.jl"))
using Random

const RES = joinpath(@__DIR__, "results")
mkpath(RES)
const KEYS = ["pair", "mixed"]
const EPS = [0.0, 1.0e-2, 1.0e-1]
const SIZES = [2, 4, 8, 16, 32, 64, 128, 320]
const SEEDS = [collect(1:32); collect(2001:2288)]   # the training seeds of split 1, extended without its test seeds

# additive noise of uniform direction and magnitude in [0, ε]
function perturb(A, ε, rng)
    B = copy(A)
    ε == 0 && return B
    for i in axes(A, 2), t in axes(A, 3)
        n = randn(rng, size(A, 1))
        B[:, i, t] .+= ε * rand(rng) .* n ./ norm(n)
    end
    return B
end

function samples(rule, seeds, ε, rng)
    envs, accs = Environment{Float64}[], SVector{2, Float64}[]
    for s in seeds
        tr = stable_simulate(rule; SIM..., speeds = initial_speeds(rule), nsteps = SAMPLING.nsteps, seed = s)
        e, a = trajectory_data(perturb(tr.pos, ε, rng), perturb(tr.vel, ε, rng), tr.dt; cutoff = interaction_radius(rule),
            extent = tr.extent, periodic = true, times = 1:SAMPLING.every:SAMPLING.nsteps)
        append!(envs, e); append!(accs, a)
    end
    return envs, accs
end

relerr(p, a) = sqrt(sum(norm.(p - a) .^ 2) / sum(norm.(a) .^ 2))

out = joinpath(RES, "efficiency.jsonl")
done = isfile(out) ? Set((String(match(r"\"swarm\": \"([^\"]*)\"", l).captures[1]), parse(Float64, match(r"\"eps\": ([-+0-9.eE]+)", l).captures[1]),
    parse(Int, match(r"\"sims\": (\d+)", l).captures[1])) for l in eachline(out)) : Set()
for key in KEYS
    rule = only(s.rule for s in SWARMS if s.key == key)
    te, tacc = samples(rule, split_seeds(1)[2], 0.0, Xoshiro(0))
    for ε in EPS, n in SIZES
        (key, Float64(ε), n) in done && continue
        envs, accs = samples(rule, SEEDS[1:n], ε, Xoshiro(1))
        b = dACEBasis(maxdeg = DEGREES["d4"], maxl = 0, maxm = 1)
        t = @elapsed m = fitcv(b, envs, accs)
        err = relerr(predict(m, te), tacc)
        open(out, "a") do io
            @printf(io, "{\"swarm\": \"%s\", \"eps\": %g, \"sims\": %d, \"train\": %d, \"relerr\": %.6g, \"lambda\": %g, \"fit_s\": %.1f}\n",
                key, ε, n, length(envs), err, m.λ, t)
        end
        @printf("%-6s ε %-5g %3d simulations (%6d samples): relative error %.4g (λ %g, %.0f s)\n", key, ε, n, length(envs), err, m.λ, t)
        flush(stdout)
    end
    checkpoint("efficiency $key")
end
