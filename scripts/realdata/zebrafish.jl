# Zebrafish (Heras et al. 2019, https://doi.org/10.1371/journal.pcbi.1007354): the idtracker.ai trajectories of
# groups of 60, 80 and 100 fish, as DATA/zebrafish/<video>.h5 (positions (2, fish, frames) in px, attributes
# body_length in px and frames_per_second; convert_zebrafish.py writes them from the published .npy files) and,
# for the cross-tracking test, the original 2019 tracking of the same videos in DATA/zebrafish2019/ (missing
# positions NaN; read with the body length and frame rate of the 2023 tracking, so that only the tracking differs).

isdefined(@__MODULE__, :SIM) || include(joinpath(@__DIR__, "..", "paper", "setup.jl"))
using HDF5

const ZEBRAFISH_DIR = joinpath(DATA, "zebrafish")
const ZEBRAFISH_DIR2019 = joinpath(DATA, "zebrafish2019")
const ZEBRAFISH_VIDEOS = ["zebrafish_60_1", "zebrafish_60_2", "zebrafish_60_3", "zebrafish_80_1", "zebrafish_80_2",
    "zebrafish_80_3", "zebrafish_100_2", "zebrafish_100_3"]   # 100-fish video 1 is not in the 2023 tracking

"""
    load_zebrafish(video; dir = ZEBRAFISH_DIR) -> (X, dt, centre, radius)

Positions in BL as `(2, fish, frames)`, the frame interval and the arena circle (centre, radius) in BL,
estimated from all positions.
"""
function load_zebrafish(video; dir = ZEBRAFISH_DIR)
    bl, fps = h5open(h -> (read_attribute(h, "body_length"), read_attribute(h, "frames_per_second")), joinpath(ZEBRAFISH_DIR, "$video.h5"))
    h5open(joinpath(dir, "$video.h5")) do h
        X = read(h["positions"]) ./ bl
        dir == ZEBRAFISH_DIR && !all(isfinite, X) && error("$video has missing positions")
        ok = [all(isfinite, view(X, :, i, t)) for i in axes(X, 2), t in axes(X, 3)]
        xs, ys = X[1, :, :][ok], X[2, :, :][ok]
        c = SVector((minimum(xs) + maximum(xs)) / 2, (minimum(ys) + maximum(ys)) / 2)
        return X, 1 / fps, c, maximum(norm(SVector(x, y) - c) for (x, y) in zip(xs, ys))
    end
end

"""
    zebrafish_samples(video; R = 8.0, gap = 0, dir = ZEBRAFISH_DIR) -> (; envs, accs, keys, speed, Δ, amax, ncoincident)

The samples of the leakage-free protocol, as `shiner_samples`: steps of `K = 3` frames, velocities
`(x(t) - x(t-K)) / Δ`, neighbours within `R` BL, the focal's past velocities at lags `1:4`, the target
`(x(u+2K) - 2x(u+K) + x(u)) / Δ²` from `u = t + 1 + gap` on. Fish at least 2 BL inside the arena, tracked over
every frame the sample uses, frames subsampled to about 50 000 samples per video; samples with accelerations
above the 99.9% quantile of the video (tracking glitches) or a neighbour at distance 0 (collapsed identities)
dropped. `keys`: the (frame, fish) of each sample; `speed`: the median and 95% quantile of the speeds.
"""
function zebrafish_samples(video; R = 8.0, gap = 0, dir = ZEBRAFISH_DIR)
    K, H, WALL, TARGET, AQ = 3, 4, 2.0, 50_000, 0.999
    X, dt, c, rad = load_zebrafish(video; dir)
    Δ = K * dt
    n, T = size(X, 2), size(X, 3)
    P(t, i) = SVector(X[1, i, t], X[2, i, t])
    vel(t, j) = (P(t, j) - P(t - K, j)) / Δ
    acc(t, i) = (u = t + 1 + gap; (P(u + 2K, i) - 2P(u + K, i) + P(u, i)) / Δ^2)
    seen(t, j) = all(isfinite, P(t, j)) && all(isfinite, P(t - K, j))
    function env(t, i)
        near = [j for j in 1:n if j != i && seen(t, j) && norm(P(t, j) - P(t, i)) < R]
        return Environment(vel(t, i), [P(t, j) - P(t, i) for j in near], [vel(t, j) - vel(t, i) for j in near];
            past_v = [vel(t - k * K, i) for k in 1:H], dt = Δ)
    end
    stride = max(K, round(Int, T * n / TARGET))
    ts = ((H + 1) * K + 1):stride:(T - 2K - 1 - gap)
    tracked(t, i) = all(u -> all(isfinite, P(u, i)), (t - (H + 1) * K):(t + 1 + 2K + gap))
    cand = [(t, i) for t in ts for i in 1:n if tracked(t, i) && rad - norm(P(t, i) - c) >= WALL]
    amax = quantile([norm(acc(t, i)) for (t, i) in cand], AQ)
    coincident(t, i) = any(j != i && norm(P(t, j) - P(t, i)) <= 1.0e-6 for j in 1:n)
    # the previous target (frames t - K + 1 onwards) is screened too, so that the samples match the AR(1) runs
    keep = [(t, i) for (t, i) in cand if norm(acc(t, i)) <= amax && norm(acc(t - K, i)) <= amax && !coincident(t, i) && !coincident(t - K, i)]
    envs = Vector{Environment{Float64}}(undef, length(keep))
    Threads.@threads for s in eachindex(keep)
        envs[s] = env(keep[s]...)
    end
    speeds = [norm(e.v) for e in envs]
    return (; envs, accs = [acc(t, i) for (t, i) in keep], keys = keep, speed = (median(speeds), quantile(speeds, 0.95)), Δ,
        amax, ncoincident = count(((t, i),) -> norm(acc(t, i)) <= amax && norm(acc(t - K, i)) <= amax, cand) - length(keep))
end
