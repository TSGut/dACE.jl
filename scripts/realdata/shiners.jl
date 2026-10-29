# Golden shiners (Davidson et al. 2021, https://doi.org/10.1098/rsif.2021.0142): the Dryad trajectories
# (https://doi.org/10.5061/dryad.sbcc2fr2h) compacted once to DATA/shiners/<trial>.h5, and the samples of the
# leakage-free protocol of the paper (shiners_protocol.jl, consistency.jl).

isdefined(@__MODULE__, :SIM) || include(joinpath(@__DIR__, "..", "paper", "setup.jl"))
using HDF5

const SHINER_RAW = joinpath(DATA, "shiners_dryad")   # the unpacked Dryad folder: <trial>/*_fov.h5
const SHINER_DIR = joinpath(DATA, "shiners")
const SHINER_TRIALS = ["0066", "0105", "0126", "0084", "0115", "0120", "0103", "0107", "0124", "150-fish"]
const SHINER_DT = 1 / 29.97

"""
    compact_shiners()

Positions, headings and body lengths (pixels) of every trial, one segment per raw file
(fish indices restart in each file), written to `SHINER_DIR/<trial>.h5`.
"""
function compact_shiners()
    mkpath(SHINER_DIR)
    for trial in SHINER_TRIALS
        out = joinpath(SHINER_DIR, "$trial.h5")
        isfile(out) && continue
        files = sort(filter(endswith("_fov.h5"), readdir(joinpath(SHINER_RAW, trial))))
        h5open(out, "w") do o
            for (s, f) in enumerate(files)
                h5open(joinpath(SHINER_RAW, trial, f)) do h
                    F = h["fields"]
                    for k in ("x", "y", "heading_x", "heading_y", "body_length")
                        o["segment$s/$k"] = read(F[k])
                    end
                end
            end
        end
        println("compacted $trial: ", length(files), " segments")
    end
end

"""
    load_shiners(trial) -> (segments, body_length)

Positions in body lengths (median body length of the trial, in pixels), one named tuple `(x, y)` of
frame-by-fish matrices per segment.
"""
function load_shiners(trial)
    h5open(joinpath(SHINER_DIR, "$trial.h5")) do h
        segs = sort(keys(h); by = s -> parse(Int, s[8:end]))
        bl = median(filter(isfinite, reduce(vcat, [vec(read(h[s]["body_length"])) for s in segs])))
        return [(x = read(h[s]["x"]) ./ bl, y = read(h[s]["y"]) ./ bl) for s in segs], bl
    end
end

"""
    shiner_samples(trial; R = 3.0, gap = 0, keys = false) -> (envs, accs[, keys])

The samples of the leakage-free protocol: inputs from frames up to `t` (velocities `(x(t) - x(t-K)) / Δ`
with `K = 3` frames, neighbours within `R` BL, the focal's past velocities at lags `1:H`, `H = 4`), the target
`(x(u+2K) - 2x(u+K) + x(u)) / Δ²` from `u = t + 1 + gap` on. Fish at least 2 BL inside the bounding box of all
positions, tracked over every frame a sample uses (for any gap up to 6, so the samples are the same at every
gap), frames subsampled to about 25 000 samples per trial; samples with accelerations above 15 BL/s² or
speeds above the 99.9% quantile of the trial (tracking glitches) dropped. `keys = true` also returns the
(segment, frame, fish) of each sample.
"""
function shiner_samples(trial; R = 3.0, gap = 0, keys = false)
    K, H, GMAX, WALL, TARGET, AMAX, VQ = 3, 4, 6, 2.0, 25_000, 15.0, 0.999
    segs, _ = load_shiners(trial)
    xs = reduce(vcat, [vec(s.x) for s in segs]); ys = reduce(vcat, [vec(s.y) for s in segs])
    box = (minimum(filter(isfinite, xs)), maximum(filter(isfinite, xs)), minimum(filter(isfinite, ys)), maximum(filter(isfinite, ys)))
    Δ = K * SHINER_DT
    n = size(segs[1].x, 2)
    stride = max(K, round(Int, sum(size(s.x, 1) for s in segs) * n / TARGET))
    envs, accs, ks = Environment{Float64}[], SVector{2, Float64}[], NTuple{3, Int}[]
    inside(p) = min(p[1] - box[1], box[2] - p[1], p[2] - box[3], box[4] - p[2]) >= WALL
    for (si, s) in enumerate(segs)
        P(t, i) = SVector(s.x[t, i], s.y[t, i])
        T = size(s.x, 1)
        vel(t, j) = (P(t, j) - P(t - K, j)) / Δ
        acc(t, i) = (u = t + 1 + gap; (P(u + 2K, i) - 2P(u + K, i) + P(u, i)) / Δ^2)
        tracked(t, i) = all(u -> all(isfinite, P(u, i)), (t - (H + 1) * K):(t + 1 + 2K + GMAX))
        for t in ((H + 1) * K + 1):stride:(T - 2K - 1 - GMAX), i in 1:n
            inside(P(t, i)) && tracked(t, i) || continue
            a = acc(t, i)
            norm(a) <= AMAX && all(isfinite, a) || continue
            near = [j for j in 1:n if j != i && all(isfinite, P(t, j)) && all(isfinite, P(t - K, j)) && norm(P(t, j) - P(t, i)) < R]
            push!(envs, Environment(vel(t, i), [P(t, j) - P(t, i) for j in near], [vel(t, j) - vel(t, i) for j in near];
                past_v = [vel(t - k * K, i) for k in 1:H], dt = Δ))
            push!(accs, a)
            push!(ks, (si, t, i))
        end
    end
    vq = quantile([norm(e.v) for e in envs], VQ)
    ok = [norm(e.v) <= vq && all(u -> norm(u) <= vq, e.past_v) for e in envs]
    return keys ? (envs[ok], accs[ok], ks[ok]) : (envs[ok], accs[ok])
end
