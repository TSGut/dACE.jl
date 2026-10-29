# julia --project=scripts/paper scripts/paper/fig_data.jl [outdir]
#
# Figure 2, the benchmark data (drawn by plots.jl, one PDF per panel): one data point of the 3-body collective, and
# the number of neighbours per data point of every collective in 2D and 3D.

isdefined(@__MODULE__, :SIM) || include(joinpath(@__DIR__, "setup.jl"))

let
    # (a) one data point of the 3-body collective in 2D: a typical neighbour count, and among those the least
    # crowded environment (largest smallest separation)
    rule = only(s.rule for s in SWARMS if s.key == "threebody")
    full = stable_simulate(rule; SIM..., speeds = initial_speeds(rule), nsteps = SAMPLING.nsteps, seed = first(TRAIN_SEEDS))
    t = 31
    envs = environments(full.pos[:, :, t], full.vel[:, :, t]; cutoff = rule.radius, extent = full.extent, periodic = true)
    sep(e) = minimum((norm(p - q) for p in [[zero(e.v)]; e.r] for q in e.r if p !== q); init = Inf)
    typical = findall(e -> abs(length(e) - median(length.(envs))) <= 2, envs)
    i = typical[argmax(sep.(envs[typical]))]
    e = envs[i]
    a = SVector{2}(full.vel[:, i, t + 1] - full.vel[:, i, t]) / full.dt
    # (b) the neighbours per data point of the training data
    plotfig("data", (; labels = [s.label for s in SWARMS], e_r = collect(e.r), e_v = e.v, e_vj = [u + e.v for u in e.u], a, R = rule.radius,
        n2 = [length.(swarm_data(s.rule, TRAIN_SEEDS)[1]) for s in SWARMS], n3 = [length.(swarm_data3(s.rule, TRAIN_SEEDS)[1]) for s in SWARMS]))
end
