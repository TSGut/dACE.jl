# julia -t 5 --project=scripts/paper scripts/synthetic/stochastic.jl
#
# Stochastic self-propulsion (Section 3.4, Figure 9): the pairwise and 2+3+4-body rules with a time-correlated
# drive per agent (ou: Ornstein-Uhlenbeck; burst: burst-and-coast), so the observed acceleration is F + d and F is
# known. 4-body fits scored on driven test simulations by the next-step R² and the relative error against F:
# (a) with and without the past acceleration as an input, (b) the self term fitted jointly and in stages.
# out: results/stochastic.jsonl (a), results/stochastic_self.jsonl (b), one line per data set

include(joinpath(@__DIR__, "..", "paper", "setup.jl"))
using Random

const RES = joinpath(@__DIR__, "results")
mkpath(RES)
const KEYS = ["pair", "mixed"]
const DRIVES = ["ou", "burst"]
const SPLITS = 1:3
const TAU, PERIOD, PHASE_NOISE = 0.5, 1.0, 0.5
const STEPS = 3:2:60

minimage(dx, L) = dx .- L .* round.(dx ./ L)
function neighbours_of(X, V, i, R, L)
    out = Tuple{SVector{2, Float64}, SVector{2, Float64}}[]
    for j in eachindex(X)
        j == i && continue
        d = minimage(X[j] - X[i], L)
        norm(d) < R && push!(out, (d, V[j]))
    end
    return out
end

# positions, velocities (2, N, nsteps + 1) and the true interaction force F (2, N, nsteps) at each step
function simulate_driven(rule, drive, amp; seed, nsteps = 60)
    n, L, dt, R = SIM.n_agents, SIM.extent[1], SIM.dt, interaction_radius(rule)
    sp = initial_speeds(rule)
    for attempt in 0:99
        rng = Xoshiro(seed + 1_000_000attempt)
        X = [SVector(L / 2, L / 2) + (rand(rng, SVector{2}) .- 0.5) .* SIM.spread for _ in 1:n]
        V = [(sp[1] + (sp[2] - sp[1]) * rand(rng)) * (θ = 2π * rand(rng); SVector(cos(θ), sin(θ))) for _ in 1:n]
        ξp, ξn, φ = zeros(n), zeros(n), 2π .* rand(rng, n)
        if amp > 0   # start the drive in its stationary state
            ξp .= amp .* randn(rng, n); ξn .= amp .* randn(rng, n)
        end
        pos, vel, F = zeros(2, n, nsteps + 1), zeros(2, n, nsteps + 1), zeros(2, n, nsteps)
        ok = true
        for t in 1:(nsteps + 1)
            for i in 1:n
                pos[:, i, t], vel[:, i, t] = X[i], V[i]
            end
            t == nsteps + 1 && break
            Vn = similar(V)
            for i in 1:n
                f = dACESwarms.acceleration(rule, V[i], neighbours_of(X, V, i, R, L))
                F[:, i, t] = f
                e = norm(V[i]) > 0 ? V[i] / norm(V[i]) : SVector(1.0, 0.0)
                nh = SVector(-e[2], e[1])
                c = exp(-dt / TAU)
                ξn[i] = c * ξn[i] + amp * sqrt(1 - c^2) * randn(rng)
                if drive == "ou"
                    ξp[i] = c * ξp[i] + amp * sqrt(1 - c^2) * randn(rng)
                    d = ξp[i] * e + ξn[i] * nh
                else
                    φ[i] += 2π / PERIOD * dt + PHASE_NOISE * sqrt(dt) * randn(rng)
                    d = amp * 16 / sqrt(26) * (max(0.0, sin(φ[i]))^4 - 3 / 16) * e + ξn[i] * nh
                end
                Vn[i] = V[i] + dt * (f + d)
            end
            if !all(v -> all(isfinite, v) && norm(v) < 1.0e3, Vn)
                ok = false
                break
            end
            V = Vn
            X = [mod.(X[i] + dt * V[i], L) for i in 1:n]
        end
        ok && return (; pos, vel, F, dt, extent = (L, L))
    end
    return error("no stable driven simulation for seed $seed")
end

# environments with the past velocities of `history`, observed accelerations and the true force at the same samples
function samples(rule, drive, amp, seeds; history)
    envs, accs, F = Environment{Float64}[], SVector{2, Float64}[], SVector{2, Float64}[]
    for s in seeds
        tr = simulate_driven(rule, drive, amp; seed = s)
        e, a = trajectory_data(tr.pos, tr.vel, tr.dt; cutoff = interaction_radius(rule), extent = tr.extent, periodic = true, times = STEPS, history)
        append!(envs, e); append!(accs, a)
        kept = filter(t -> t > dACE.history_length(history), collect(STEPS))   # the steps with the history
        append!(F, [SVector(tr.F[1, i, t], tr.F[2, i, t]) for t in kept for i in 1:SIM.n_agents])
    end
    return envs, accs, F
end

relerr(p, a) = sqrt(sum(norm.(p - a) .^ 2) / sum(norm.(a) .^ 2))
R2(p, a) = 1 - sum(norm.(p - a) .^ 2) / sum(norm.(a) .^ 2)

function amplitude(rule)
    tr = simulate_driven(rule, "ou", 0.0; seed = 1)
    return sqrt(sum(abs2, tr.F) / (size(tr.F, 2) * size(tr.F, 3)))
end

# minimal JSON for nested Dicts of numbers and strings
json(x::AbstractString) = "\"$x\""
json(x::Real) = isfinite(x) ? @sprintf("%.6g", x) : "null"
json(x::AbstractDict) = "{" * join(["\"$k\": $(json(v))" for (k, v) in x], ", ") * "}"

for key in KEYS
    rule = only(s.rule for s in SWARMS if s.key == key)
    amp = amplitude(rule)
    @printf("%s: drive amplitude %.3f (RMS interaction force of the undriven rule)\n", key, amp)
    for drive in DRIVES, r in SPLITS
        train_seeds, test_seeds = split_seeds(r)
        b = dACEBasis(maxdeg = DEGREES["d4"], maxl = 0, maxm = 1)
        head = Dict{String, Any}("swarm" => key, "drive" => drive, "split" => r, "amp" => amp)

        # (a) the past acceleration as an input: no past against the past acceleration in the self term
        past1 = History(selfterm = 1, use = :acceleration, selfterm_degree = (6, 0))
        envs, accs, _ = samples(rule, drive, amp, train_seeds; history = past1)
        te, ta, tF = samples(rule, drive, amp, test_seeds; history = past1)
        m0 = fitcv(b, envs, accs)
        p0 = predict(m0, te)
        # the interaction degrees of the 4-body model, a higher 1-body degree for the self term
        bm = dACEBasis(maxdeg = (8, DEGREES["d4"][2:end]...), maxl = 0, maxm = 1, history = past1,
            ascale = 2median(norm(dACE.own_past(e, 1, :acceleration)) for e in envs))
        mm = fit(bm, envs, accs; prior = prior_of(bm), reflections = true, self_fit = :joint)
        pm = predict(mm, te)
        out = merge(head, Dict("train" => length(envs),
            "iid" => Dict("onestep_R2" => R2(p0, ta), "force_relerr" => relerr(p0, tF), "lambda" => m0.λ),
            "memory_onebody" => Dict("onestep_R2" => R2(pm, ta), "force_relerr" => relerr(pm, tF), "lambda" => mm.λ, "basis" => length(bm))))
        open(io -> println(io, json(out)), joinpath(RES, "stochastic.jsonl"), "a")
        @printf("%-6s %-5s split %d: next-step R² no past %.3f, past acceleration %.3f | force error %.3g, %.3g\n",
            key, drive, r, out["iid"]["onestep_R2"], out["memory_onebody"]["onestep_R2"], out["iid"]["force_relerr"], out["memory_onebody"]["force_relerr"])

        # (b) the self term of the velocities 1-4 steps back, joint and staged
        memory = History(selfterm = 1:4, selfterm_degree = (8, 4))
        envs, accs, _ = samples(rule, drive, amp, train_seeds; history = memory)
        te, ta, tF = samples(rule, drive, amp, test_seeds; history = memory)
        truesocial = [f - (rule.α - rule.β * dot(x.v, x.v)) * x.v for (f, x) in zip(tF, te)]
        social(model, x) = sum(body_order_contributions(model, x)[2:end])
        m0 = fitcv(b, envs, accs)   # memoryless, on the same samples
        bs = dACEBasis(maxdeg = DEGREES["d4"], maxl = 0, maxm = 1, history = memory)
        mj = fit(bs, envs, accs; prior = prior_of(bs), reflections = true, self_fit = :joint)
        ms = fit(bs, envs, accs; prior = prior_of(bs), reflections = true, self_fit = :staged)
        K = length(bs.funcs)
        interaction = dACEModel(b, mj.c[1:K])   # the interaction columns of the joint fit
        out = merge(head, Dict("train" => length(envs), "basis" => length(bs),
            "iid" => Dict("onestep_R2" => R2(predict(m0, te), ta), "force_relerr" => relerr(predict(m0, te), tF),
                "social_relerr" => relerr([social(m0, x) for x in te], truesocial), "lambda" => m0.λ),
            "staged" => Dict("onestep_R2" => R2(predict(ms, te), ta), "social_relerr" => relerr([social(ms, x) for x in te], truesocial),
                "lambda" => ms.λ, "lambda_self" => ms.λself),
            "self4" => Dict("onestep_R2" => R2(predict(mj, te), ta), "force_relerr" => relerr(predict(interaction, te), tF),
                "social_relerr" => relerr([social(mj, x) for x in te], truesocial), "lambda" => mj.λ)))
        open(io -> println(io, json(out)), joinpath(RES, "stochastic_self.jsonl"), "a")
        @printf("%-6s %-5s split %d: next-step R² joint %.3f staged %.3f no self term %.3f | social-part error joint %.3g, staged %.3g\n",
            key, drive, r, out["self4"]["onestep_R2"], out["staged"]["onestep_R2"], out["iid"]["onestep_R2"], out["self4"]["social_relerr"], out["staged"]["social_relerr"])
        flush(stdout)
    end
    checkpoint("stochastic $key")
end
