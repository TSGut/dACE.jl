# shared setup of the paper scripts

using dACE, dACESwarms
using CairoMakie, LinearAlgebra, Printf, Serialization, Statistics, StaticArrays

const OUT = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "figures")   # figures, numbers and cached plot data
mkpath(OUT)
const DATA = get(ENV, "DACE_DATA", joinpath(@__DIR__, "..", "..", "data"))   # tracking data (scripts/realdata)

# optional resource log of pipeline stages (JSON lines): set DACE_RUNLOG to a file path. `checkpoint(stage)`
# records the wall time, allocations and GC time since the previous checkpoint and the peak memory so far
const RUNLOG = get(ENV, "DACE_RUNLOG", "")
const LAST_CHECKPOINT = Ref((time(), Base.gc_bytes(), Base.gc_time_ns()))
function checkpoint(stage)
    isempty(RUNLOG) && return nothing
    t, b, g = time(), Base.gc_bytes(), Base.gc_time_ns()
    t0, b0, g0 = LAST_CHECKPOINT[]
    LAST_CHECKPOINT[] = (t, b, g)
    open(RUNLOG, "a") do io
        @printf(io, "{\"stage\": \"%s\", \"script\": \"%s\", \"end\": \"%s\", \"wall_s\": %.1f, \"alloc_GiB\": %.2f, \"gc_s\": %.1f, \"maxrss_GiB\": %.2f, \"threads\": %d}\n",
            stage, basename(PROGRAM_FILE), Libc.strftime("%Y-%m-%dT%H:%M:%S", t), t - t0, (b - b0) / 2^30, (g - g0) / 1.0e9, Sys.maxrss() / 2^30, Threads.nthreads())
    end
    return nothing
end

# ---------------------------------------------------------------- data

const SIM = (; n_agents = 30, extent = (100.0, 100.0), spread = 12.0, speeds = (0.2, 2.5), dt = 0.05)
const TRAIN_SEEDS, TEST_SEEDS = 1:32, 101:104
const SKIPPED = Int[]  # resampled unstable seeds

# first stable simulation among seed, seed + 10⁶, seed + 2·10⁶, …
function stable_simulate(rule; seed, kwargs...)
    for k in 0:99
        try
            return simulate(rule; kwargs..., seed = seed + 1_000_000k)
        catch err
            occursin("unstable", sprint(showerror, err)) || rethrow()
            push!(SKIPPED, seed + 1_000_000k)
        end
    end
    return error("no stable simulation for seed $seed")
end

# constant-speed rules start at their speed
initial_speeds(rule) = rule isa Union{Boids, Zonal} ? (rule.speed, rule.speed) : SIM.speeds

# frames simulated per seed and the sampling interval of the training and test data
const SAMPLING = (nsteps = 60, every = 2)

function swarm_data(rule, seeds; nsteps = SAMPLING.nsteps, every = SAMPLING.every, cutoff = interaction_radius(rule))
    envs, accs = Environment{Float64}[], SVector{2, Float64}[]
    for seed in seeds
        tr = stable_simulate(rule; SIM..., speeds = initial_speeds(rule), nsteps, seed)
        e, a = trajectory_data(
            tr.pos, tr.vel, tr.dt; cutoff, extent = tr.extent,
            periodic = true, times = 1:every:nsteps,
        )
        append!(envs, e)
        append!(accs, a)
    end
    return envs, accs
end

const SWARMS = [
    (key = "pair", label = "pairwise", rule = DOrsogna()),
    (key = "threebody", label = "3-body flanking", rule = DOrsogna(κv = 2.0)),
    (key = "averaged", label = "averaged alignment", rule = DOrsogna(averaged = true)),
    (key = "mixed", label = "2+3+4-body", rule = DOrsogna(κv = 2.0, κ4 = 4.0)),
    # discontinuous rules
    (key = "boids", label = "Boids", rule = Boids()),
    (key = "zonal", label = "zonal", rule = Zonal()),
]
# chiral variants (2D only): the pair forces turned by a fixed angle, a constant turning rate, and alignment
# weighted towards neighbours on the left (dACESwarms.DOrsogna)
const CHIRAL_SWARMS = [
    (key = "pair_turned10", label = "pairwise, turned 10°", rule = DOrsogna(chirality = deg2rad(10))),
    (key = "pair_turned30", label = "pairwise, turned 30°", rule = DOrsogna(chirality = deg2rad(30))),
    (key = "pair_turning", label = "pairwise, turning", rule = DOrsogna(ω = 0.3)),
    (key = "averaged_lateral03", label = "averaged, lateral 0.3", rule = DOrsogna(averaged = true, lateral = 0.3)),
    (key = "averaged_lateral07", label = "averaged, lateral 0.7", rule = DOrsogna(averaged = true, lateral = 0.7)),
]

# total degree per body order of each model, set by hand as in ACE: one setting per model for every collective,
# nested (each model keeps the degrees of the smaller ones) and decreasing from the 2-body term on; λ is chosen by
# the evidence in every fit
const DEGREES = Dict("d2" => (6, 12), "d3" => (6, 12, 8), "d4" => (6, 12, 8, 6))
const DEGREES3 = Dict("d2" => (6, 12), "d3" => (6, 12, 7), "d4" => (6, 12, 7, 5))
const MODELS = [
    (key = "d2", label = "dACE 2-body", basis = () -> dACEBasis(maxdeg = DEGREES["d2"], maxl = 0, maxm = 1)),
    (key = "d3", label = "dACE 3-body", basis = () -> dACEBasis(maxdeg = DEGREES["d3"], maxl = 0, maxm = 1)),
    (key = "d4", label = "dACE 4-body", basis = () -> dACEBasis(maxdeg = DEGREES["d4"], maxl = 0, maxm = 1)),
]
const MODELS3 = [
    (key = "d2", label = "dACE 2-body", bases = () -> heading_bases(maxdeg = DEGREES3["d2"], maxa = 1)),
    (key = "d3", label = "dACE 3-body", bases = () -> heading_bases(maxdeg = DEGREES3["d3"], maxa = 1)),
    (key = "d4", label = "dACE 4-body", bases = () -> heading_bases(maxdeg = DEGREES3["d4"], maxa = 1)),
]

# ---------------------------------------------------------------- style

const INK, INK2, GRID = colorant"#0b0b0b", colorant"#52514e", colorant"#e4e3df"
const COLOR = Dict(
    # dACE by body order (2 yellow, 3 blue, 4 forest green) and a neutral grey
    "d2" => colorant"#eda100", "d3" => colorant"#1f63b8", "d4" => colorant"#33a02c", "grey" => colorant"#a3a19c",
)
const FORCE = colorant"#eb6834"    # force and acceleration arrows in the schematics
const PURPLE = colorant"#7b4fa0"   # a second dACE variant beside its model colour (joint self term, relative velocity)
const LINESTYLE = Dict("d2" => :dash, "d3" => :solid, "d4" => :dot)

# text sizes of all figures (pt at the printed size: each PDF is drawn at its width in the paper)
const TICK, LABEL, LEGEND, TITLE, ANNOT = 8, 9, 8, 9, 8

set_theme!(
    Theme(
        fontsize = 9, textcolor = INK, backgroundcolor = :white, linewidth = 2,
        Axis = (
            xgridcolor = GRID, ygridcolor = GRID, xgridwidth = 0.75, ygridwidth = 0.75,
            leftspinecolor = INK2, bottomspinecolor = INK2, topspinevisible = false,
            rightspinevisible = false, xtickcolor = INK2, ytickcolor = INK2,
            xticklabelcolor = INK2, yticklabelcolor = INK2, titlefont = :regular,
            titlealign = :left, titlesize = TITLE, xticklabelsize = TICK, yticklabelsize = TICK,
            xlabelsize = LABEL, ylabelsize = LABEL,
        ),
        Axis3 = (xticklabelsize = TICK, yticklabelsize = TICK, zticklabelsize = TICK),
        Legend = (framevisible = false, labelsize = LEGEND),
    )
)

const WIDTH = 455  # pt, text width


const SIM3 = (; n_agents = 40, extent = (100.0, 100.0, 100.0), spread = 10.0, speeds = (0.2, 2.5), dt = 0.05)

function swarm_data3(rule, seeds; nsteps = SAMPLING.nsteps, every = SAMPLING.every)
    envs, accs = Environment3{Float64}[], SVector{3, Float64}[]
    for seed in seeds
        tr3 = stable_simulate(rule; SIM3..., speeds = initial_speeds(rule), nsteps, seed)
        e, a = trajectory_data(
            tr3.pos, tr3.vel, tr3.dt; cutoff = interaction_radius(rule), extent = tr3.extent,
            periodic = true, times = 1:every:nsteps,
        )
        append!(envs, e)
        append!(accs, a)
    end
    return envs, accs
end


# ---------------------------------------------------------------- trajectory helpers

# NaN-separated paths, broken at periodic wraps
function paths(pos; upto = size(pos, 3), from = 1, stride = 1)
    out = Point2f[]
    for i in axes(pos, 2)
        steps = unique([from:stride:upto; upto])
        for (k, s) in enumerate(steps)
            k > 1 && norm(pos[:, i, s] - pos[:, i, steps[k - 1]]) > 10 && push!(out, Point2f(NaN, NaN))
            push!(out, Point2f(pos[1, i, s], pos[2, i, s]))
        end
        push!(out, Point2f(NaN, NaN))
    end
    return out
end

function square_limits(trajectories...; pad = 0.06)
    xs = reduce(vcat, [vec(P[1, :, :]) for P in trajectories])
    ys = reduce(vcat, [vec(P[2, :, :]) for P in trajectories])
    c = ((minimum(xs) + maximum(xs)) / 2, (minimum(ys) + maximum(ys)) / 2)
    w = (1 + pad) * max(maximum(xs) - minimum(xs), maximum(ys) - minimum(ys)) / 2
    return (c[1] - w, c[1] + w, c[2] - w, c[2] + w)
end

heading(vel, t) = [atan(vel[2, i, t], vel[1, i, t]) for i in axes(vel, 2)]

# ridge with the algebraic smoothness prior; λ by the evidence (the default of fit), on all the training data
prior_of(basis) = basis isa Tuple ? smoothness_prior.(basis) : smoothness_prior(basis)
# real coefficients (mirror symmetric) by default, since every benchmark rule is achiral; `reflections = false` for
# chiral rules
fitcv(basis, envs, accs; reflections = true) = fit(basis, envs, accs; prior = prior_of(basis), reflections)

decades(lo, hi) = 10.0 .^ (floor(Int, log10(lo)):ceil(Int, log10(hi)))
# log axes label whole decades, 10ⁿ (every second decade where the range is wide)
decades2(lo, hi) = (d = decades(lo, hi); length(d) > 6 ? d[(mod(round(Int, log10(d[1])), 2) + 1):2:end] : d)
const DECADE_FORMAT = vs -> [rich("10", superscript(string(round(Int, log10(v))))) for v in vs]

# two significant digits for tables and macros, powers of ten written out for small values
function fmt2(x)
    isinf(x) && return x > 0 ? "\$\\infty\$" : "\$-\\infty\$"   # e.g. a diverged rollout
    t = @sprintf("%.2g", x)
    occursin('e', t) || return t
    m, e = split(t, 'e')
    return "\$" * (m == "1" ? "" : m * " \\times ") * "10^{" * string(parse(Int, e)) * "}\$"
end

model_label(key) = MODELS[findfirst(m -> m.key == key, MODELS)].label

# one square panel of paths: layers of (positions, velocities, colour, alpha), headings drawn at the end
function paths_panel(file, lims, layers; width)
    fig = Figure(size = (width, width), figure_padding = 2)
    ax = Axis(fig[1, 1]; aspect = AxisAspect(1), limits = lims)
    hidedecorations!(ax)
    hidespines!(ax)
    for (pos, vel, colour, alpha) in layers
        lines!(ax, paths(pos); color = (colour, alpha), linewidth = 0.75)
        scatter!(ax, pos[1, :, end], pos[2, :, end]; marker = :rtriangle, rotation = heading(vel, size(vel, 3)), markersize = 5, color = colour)
    end
    return save(file, fig)
end

# curves: (time, error, colour, linestyle, label)
function error_panel(file, dt, curves; nbanks = 1, height = 115)
    fig = Figure(size = (WIDTH, height), figure_padding = 2)
    ax = Axis(fig[1, 1]; xlabel = "time", ylabel = "mean position error", yscale = log10, yticks = decades2,
        ytickformat = vs -> [rich("10", superscript(string(round(Int, log10(v))))) for v in vs])
    vspan!(ax, 0, 60 * dt; color = (GRID, 0.8))
    text!(ax, 0.01, 0.97; text = "training window", space = :relative, align = (:left, :top), fontsize = ANNOT, color = INK2)
    for (t, err, colour, ls, label) in curves
        lines!(ax, t, err; color = colour, linestyle = ls, label)
    end
    nbanks == 1 ? Legend(fig[1, 2], ax; labelsize = LEGEND, rowgap = 0, patchsize = (14, 5), padding = (4, 0, 0, 0), halign = :left) :
        Legend(fig[2, 1], ax; orientation = :horizontal, nbanks, tellheight = true, labelsize = LEGEND, rowgap = 0, patchsize = (14, 5),
            padding = (0, 0, 0, 0))
    return save(file, fig)
end

mean_error(pos, truth, L) = map(1:(size(pos, 3) - 1)) do t
    D = size(pos, 1)
    mean(norm(d - L .* round.(d ./ L)) for d in (SVector{D}(pos[:, i, t + 1] - truth[:, i, t + 1]) for i in axes(pos, 2)))
end

# trajectory figure as subfigures: one panel of paths per model (true paths grey, predicted in colour) and one panel
# of the mean position error against time, training window shaded; `project`: D × N × T → 2 × N × T for drawing.
# Writes <base>_<model>.pdf for every model and <base>_error.pdf
function rollout_figure(path, ex; project = identity, reference = nothing)
    (; models, truth, preds, nsteps) = ex
    base = replace(path, r"\.pdf$" => "")
    D = size(truth.pos, 1)
    L = SVector{D}(truth.extent)
    tpos, tvel = project(truth.pos), project(truth.vel)
    lims = square_limits(tpos, (project(preds[k][1]) for k in models)...)
    for key in models
        paths_panel("$(base)_$key.pdf", lims, ((tpos, tvel, INK2, 0.35), (project(preds[key][1]), project(preds[key][2]), COLOR[key], 0.6));
            width = 0.25WIDTH)
    end
    t = truth.dt .* (1:nsteps)
    curves = [(t, mean_error(preds[key][1], truth.pos, L)[1:nsteps], COLOR[key], LINESTYLE[key], model_label(key)) for key in models]
    isnothing(reference) || push!(curves, (t, mean_error(reference, truth.pos, L)[1:nsteps], INK2, :dot, "truth, start perturbed by 10⁻⁶"))
    return error_panel("$(base)_error.pdf", truth.dt, curves; nbanks = isnothing(reference) ? 1 : 2, height = isnothing(reference) ? 115 : 165)
end

include(joinpath(@__DIR__, "plots.jl"))

const NSPLITS = parse(Int, get(ENV, "DACE_NSPLITS", "10"))
const NSPLITS3 = parse(Int, get(ENV, "DACE_NSPLITS3", "5"))   # the 3D fits on the first NSPLITS3 splits only
split_seeds(r) = r == 1 ? (TRAIN_SEEDS, TEST_SEEDS) : (1000r .+ TRAIN_SEEDS, 1000r .+ TEST_SEEDS)
