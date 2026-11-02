# julia -t auto --project=scripts/paper scripts/paper/figures.jl [outdir]

include(joinpath(@__DIR__, "setup.jl"))
using Serialization

checkpoint("load and setup")

# ---------------------------------------------------------------- fits


println("fitting ...")
data = Dict(s.key => (train = swarm_data(s.rule, TRAIN_SEEDS), test = swarm_data(s.rule, TEST_SEEDS)) for s in SWARMS)
fit_times = Dict{Tuple{String, String}, Float64}()
fits = Dict(
    (s.key, m.key) => begin
        b = m.basis()
        fit(b, data[s.key].train[1][1:10], data[s.key].train[2][1:10]; λ = 1.0e-3)  # compile
        t = @elapsed f = fitcv(b, data[s.key].train...)
        fit_times[(s.key, m.key)] = t
        f
    end
    for s in SWARMS for m in MODELS
)
errors = Dict(k => rmse(f, data[k[1]].test...; relative = true) for (k, f) in fits)

nbrs(e) = [(e.r[j], e.u[j] + e.v) for j in 1:length(e)]
function true_orders(rule, e)
    a(r) = acceleration(r, e.v, nbrs(e))
    a0, a1, a2 = a(DOrsogna(Cr = 0.0, Ca = 0.0, γ = 0.0)), a(DOrsogna()), a(DOrsogna(κv = rule.κv))
    return [a0, a1 - a0, a2 - a1, a(rule) - a2]
end
checkpoint("fits")

# RMS error of each learned body-order contribution against the true term, relative to the RMS acceleration and
# to the true term (NaN where it vanishes)
recovery_of(m, swarm, envs, accs) = begin
    rule = only(s.rule for s in SWARMS if s.key == swarm)
    truth = [true_orders(rule, e) for e in envs]
    learned = [body_order_contributions(m, e) for e in envs]
    rms(xs) = sqrt(sum(norm.(xs) .^ 2))
    map(1:length(learned[1])) do N
        err, own = rms([l[N] - t[N] for (l, t) in zip(learned, truth)]), rms([t[N] for t in truth])
        (; acc = err / rms(accs), own = own > 1e-12 * rms(accs) ? err / own : NaN)
    end
end

# ---------------------------------------------------------------- learned kernels

# probes: one neighbour (2-body), two neighbours (3-body)
const V0 = SVector(sqrt(2.0), 0.0)
single(r, u) = Environment(V0, [r], [u])
order(m, env, N) = body_order_contributions(m, env)[N + 1]

rule = SWARMS[2].rule
m3, m2 = fits[("threebody", "d3")], fits[("threebody", "d2")]
ρmin = quantile([norm(r) for e in data["threebody"].train[1] for r in e.r], 0.01)
ρs = range(ρmin, rule.radius; length = 200)
fc(ρ) = ρ < rule.radius ? (1 - (ρ / rule.radius)^2)^2 : 0.0

δ = 0.2
φs = range(0, π; length = 200)
g(ρ) = fc(ρ) * exp(-ρ / rule.l3)
alignment_response(m) = [(order(m, single(SVector(ρ, 0.0), SVector(0.0, δ)), 1)[2] - order(m, single(SVector(ρ, 0.0), SVector(0.0, 0.0)), 1)[2]) / δ for ρ in ρs]
flank = map((1.0, 2.0)) do ρ
    pair_env(φ, u) = Environment(V0, [ρ * SVector(cos(φ / 2), sin(φ / 2)), ρ * SVector(cos(φ / 2), -sin(φ / 2))], [u, u])
    response(φ) = (order(m3, pair_env(φ, SVector(0.0, δ)), 2)[2] - order(m3, pair_env(φ, zero(V0)), 2)[2]) / δ
    (; truth = [rule.κv * g(ρ)^2 * (1 - cos(φ)) / 2 for φ in φs], learned = response.(φs))
end
plotfig("kernels", (; ρs = collect(ρs), φs = collect(φs), flank,
    true_rad = [fc(ρ) * (-rule.Cr / rule.lr * exp(-ρ / rule.lr) + rule.Ca / rule.la * exp(-ρ / rule.la)) for ρ in ρs],
    rad3 = [order(m3, single(SVector(ρ, 0.0), SVector(0.0, 0.0)), 1)[1] for ρ in ρs],
    rad2 = [order(m2, single(SVector(ρ, 0.0), SVector(0.0, 0.0)), 1)[1] for ρ in ρs],
    true_align = [rule.γ * fc(ρ) * exp(-ρ / rule.lv) for ρ in ρs], align3 = alignment_response(m3), align2 = alignment_response(m2)))

checkpoint("learned kernels")

# ---------------------------------------------------------------- trajectory prediction


"""
    rollout_experiment(swarm, models; nsteps = 400, seed = 999)

Truth and model rollouts from a held-out initial condition.
"""
function rollout_experiment(swarm, models; nsteps = 400, seed = 999)
    rule = only(s.rule for s in SWARMS if s.key == swarm)
    truth = simulate(rule; SIM..., speeds = initial_speeds(rule), nsteps, seed)
    preds = Dict(
        key => rollout(fits[(swarm, key)], truth.pos[:, :, 1], truth.vel[:, :, 1], truth.dt, nsteps; extent = truth.extent, periodic = true)
        for key in models
    )
    return (; swarm, models, truth, preds, nsteps)
end

# the 2+3+4-body rollout is Figure 4, the others are rendered as videos only (figshare, 10.6084/m9.figshare.34831176)
rollout_experiments = [rollout_experiment("mixed", ("d2", "d3", "d4")), rollout_experiment("averaged", ("d2", "d3", "d4")), rollout_experiment("threebody", ("d2", "d3"))]
plotfig("rollout_mixed", (; ex = only(ex for ex in rollout_experiments if ex.swarm == "mixed"), pca = nothing, reference = nothing))

checkpoint("trajectory prediction")

# ---------------------------------------------------------------- 3D (heading frame)

# MODELS3: the 3D models (setup.jl)
println("fitting 3D ...")
data3 = Dict(s.key => (train = swarm_data3(s.rule, TRAIN_SEEDS), test = swarm_data3(s.rule, TEST_SEEDS)) for s in SWARMS)
fits3 = Dict((s.key, m.key) => fitcv(m.bases(), data3[s.key].train...) for s in SWARMS for m in MODELS3)
errors3 = Dict(k => rmse(f, data3[k[1]].test...; relative = true) for (k, f) in fits3)

# 3D rollout, projected on the principal plane, without the 2-body model, whose rollout diverges
rollout3 = let rule = only(s.rule for s in SWARMS if s.key == "mixed"), nsteps = 400, models = ("d3", "d4")
    truth = simulate(rule; SIM3..., nsteps, seed = 999)
    preds = Dict(
        key => rollout(fits3[("mixed", key)], truth.pos[:, :, 1], truth.vel[:, :, 1], truth.dt, nsteps; extent = truth.extent, periodic = true)
        for key in models
    )
    (; swarm = "mixed3d", models, truth, preds, nsteps)
end
const PCA3 = let X = reshape(rollout3.truth.pos, 3, :)
    X = X .- mean(X; dims = 2)
    eigvecs(Symmetric(X * X'))[:, end:-1:2]'
end
project3(A) = reshape(PCA3 * reshape(A, 3, :), 2, size(A, 2), size(A, 3))
plotfig("rollout_mixed3d", (; ex = rollout3, pca = Matrix(PCA3), reference = nothing))

checkpoint("3D (heading frame)")

# ---------------------------------------------------------------- independent splits

# medians over independent splits, figures use split 1

# the self-interacting 4-body basis on the same index sets, at the degrees of the pure model
const MODEL_SI = dACEBasis(maxdeg = DEGREES["d4"], maxl = 0, maxm = 1, pure = false)

function run_split(r)
    train_seeds, test_seeds = split_seeds(r)
    e2, e3, esi = (Dict{Tuple{String, String}, Float64}() for _ in 1:3)
    rec = Dict{String, Vector{@NamedTuple{acc::Float64, own::Float64}}}()
    for s in SWARMS
        tr, te = swarm_data(s.rule, train_seeds), swarm_data(s.rule, test_seeds)
        for m in MODELS
            f = fitcv(m.basis(), tr...)
            e2[(s.key, m.key)] = rmse(f, te...; relative = true)
            (s.key in ("pair", "threebody", "mixed") && m.key == "d4") && (rec[s.key] = recovery_of(f, s.key, te...))
        end
        if s.key == "mixed"
            f = fitcv(MODEL_SI, tr...)
            esi[("mixed", "d4")] = rmse(f, te...; relative = true)
            rec["mixed_si"] = recovery_of(f, "mixed", te...)
        end
        r <= NSPLITS3 || continue   # 3D on the first NSPLITS3 splits only (its fits cost most)
        tr3, te3 = swarm_data3(s.rule, train_seeds), swarm_data3(s.rule, test_seeds)
        for m in MODELS3
            f = fitcv(m.bases(), tr3...)
            e3[(s.key, m.key)] = rmse(f, te3...; relative = true)
            (s.key, m.key) == ("mixed", "d4") && (rec["mixed3"] = recovery_of(f, "mixed", te3...))
        end
    end
    return (; errors = e2, errors3 = e3, errors_si = esi, rec)
end

# Split results are cached in OUT with the specification they were computed from (rules, seeds, settings, models,
# dACE version) and reused only when it matches. Raise `schema` when run_split changes what it computes, delete
# splits_*.jls to recompute.
split_spec(r) = (; schema = 3, with3d = r <= NSPLITS3, dace = pkgversion(dACE), swarms = [(s.key, s.rule) for s in SWARMS], seeds = split_seeds(r),
    sim = SIM, sim3 = SIM3, sampling = SAMPLING,
    models = [(m.key, DEGREES[m.key]) for m in MODELS], models3 = [(m.key, DEGREES3[m.key]) for m in MODELS3],
    fit = (λ = :evidence, prior = :smoothness))
function cached_split(r)
    spec = split_spec(r)
    path = joinpath(OUT, "splits_$(r)_$(string(hash(spec); base = 16)).jls")
    if isfile(path)
        stored = deserialize(path)
        stored isa NamedTuple && haskey(stored, :spec) && stored.spec == spec && return stored.result
        println("split $r: the cache $path was computed from other settings, recomputing")
    end
    println("split $r / $NSPLITS ...")
    flush(stdout)
    result = run_split(r)
    tmp = path * ".tmp$(getpid())"
    serialize(tmp, (; spec, result))
    mv(tmp, path; force = true)
    GC.gc()
    return result
end
splits = map(cached_split, 1:NSPLITS)
# the split results of this run, for the plotting scripts (fig_recovery.jl)
let tmp = joinpath(OUT, "splits.jls.tmp$(getpid())")
    serialize(tmp, (; nsplits3 = min(NSPLITS3, NSPLITS), splits))
    mv(tmp, joinpath(OUT, "splits.jls"); force = true)
end

write_numbers(min(NSPLITS3, NSPLITS), splits)
println("resampled unstable seeds: ", SKIPPED)

checkpoint("independent splits")

# ---------------------------------------------------------------- collective simulations for videos

snapshot(rule) = simulate(
    rule; SIM..., n_agents = 60, nsteps = 3000, seed = 7, extent = (400.0, 400.0),
    dt = 0.01, speeds = (1.0, 1.0), heading_spread = π,
)
snapshots = Dict(s.key => snapshot(s.rule) for s in SWARMS)


checkpoint("collective simulations for videos")

# ---------------------------------------------------------------- videos

const VIDEOS = joinpath(OUT, "..", "videos")
mkpath(VIDEOS)

"""
    render_video(path, panels, steps, dt; tail = nothing, stride = 1, fps = 30)

`panels = [(title, [(pos, vel, head, trail), …])]`; fixed camera.
"""
function render_video(path, panels, steps, dt; tail = nothing, stride = 1, fps = 30)
    fig = Figure(size = (420 * length(panels), 470), backgroundcolor = :white)
    t = Observable(first(steps))
    Label(fig[0, 1:length(panels)], lift(t -> @sprintf("t = %.1f", (t - 1) * dt), t); color = INK2, fontsize = 16, tellwidth = false)
    for (p, (title, layers)) in enumerate(panels)
        ax = Axis(fig[1, p]; title, titlesize = 16, aspect = AxisAspect(1), limits = square_limits(first.(layers)...))
        hidedecorations!(ax)
        hidespines!(ax)
        for (pos, vel, head, trail) in layers
            from(t) = isnothing(tail) ? 1 : max(1, t - tail)
            lines!(ax, lift(t -> paths(pos; upto = t, from = from(t), stride), t); color = (trail, 0.45), linewidth = 1.2)
            heads = lift(t -> Point2f.(pos[1, :, t], pos[2, :, t]), t)
            scatter!(ax, heads; marker = :rtriangle, rotation = lift(t -> heading(vel, t), t), markersize = 10, color = head)
        end
    end
    return record(fig, path, steps; framerate = fps) do k
        t[] = k
    end
end

println("rendering videos ...")
snapshot_layers(key) = [(snapshots[key].pos, snapshots[key].vel, INK, COLOR["d3"])]
frames = 1:10:3001
render_video(joinpath(VIDEOS, "collectives.mp4"), [(s.label, snapshot_layers(s.key)) for s in SWARMS], frames, 0.01; tail = 500, stride = 10)
for s in SWARMS
    render_video(joinpath(VIDEOS, "collective_$(s.key).mp4"), [(s.label, snapshot_layers(s.key))], frames, 0.01; tail = 500, stride = 10)
end
for ex in rollout_experiments
    render_video(
        joinpath(VIDEOS, "rollout_$(ex.swarm).mp4"),
        [
            ("$(model_label(key)) (colour) vs truth (grey)", [(ex.truth.pos, ex.truth.vel, INK2, INK2), (ex.preds[key]..., COLOR[key], COLOR[key])])
            for key in ex.models
        ],
        1:2:(ex.nsteps + 1), ex.truth.dt,
    )
end

render_video(
    joinpath(VIDEOS, "rollout_mixed3d.mp4"),
    [
        ("$(model_label(key)) (colour) vs truth (grey)", [
            (project3(rollout3.truth.pos), project3(rollout3.truth.vel), INK2, INK2),
            (project3(rollout3.preds[key][1]), project3(rollout3.preds[key][2]), COLOR[key], COLOR[key]),
        ])
        for key in rollout3.models
    ],
    1:2:(rollout3.nsteps + 1), rollout3.truth.dt,
)

for s in SWARMS, m in MODELS3
    @printf("3D %-10s %-4s %.3g%%\n", s.key, m.key, 100 * errors3[(s.key, m.key)])
end
for s in SWARMS, m in MODELS
    @printf("%-10s %-4s %.3g%%\n", s.key, m.key, 100 * errors[(s.key, m.key)])
end
println("figures written to ", abspath(OUT), ", videos to ", abspath(VIDEOS))
checkpoint("videos")

include(joinpath(@__DIR__, "fig_schematic.jl"))
checkpoint("fig_schematic")
