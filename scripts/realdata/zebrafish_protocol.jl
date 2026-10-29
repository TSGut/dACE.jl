# ZEBRAFISH_R=8 ZEBRAFISH_GAP=0 julia -t 5 --project=scripts/paper scripts/realdata/zebrafish_protocol.jl
#
# Zebrafish (Section 4): leave one video out, the 1-, 2- and 3-body laws without past and with the self term,
# scored by held-out next-step R² on the 2023 tracking and, if DATA/zebrafish2019 exists, on the 2019 tracking.
# out: results/zebrafish_R<R>_gap<G>.{jls,md}, a line in results/zebrafish_protocol.jsonl

include(joinpath(@__DIR__, "zebrafish.jl"))

const RCUT = parse(Float64, get(ENV, "ZEBRAFISH_R", "8"))
const GAP = parse(Int, get(ENV, "ZEBRAFISH_GAP", "0"))
const RES = joinpath(@__DIR__, "results")
mkpath(RES)
R2(p, a) = 1 - sum(norm.(p - a) .^ 2) / sum(norm.(a) .^ 2)

function main()
    println("zebrafish, protocol: R = $RCUT BL, gap $GAP frames, $(Threads.nthreads()) threads")
    data = Dict(v => zebrafish_samples(v; R = RCUT, gap = GAP) for v in ZEBRAFISH_VIDEOS)
    cross = isdir(ZEBRAFISH_DIR2019)
    data19 = cross ? Dict(v => zebrafish_samples(v; R = RCUT, gap = GAP, dir = ZEBRAFISH_DIR2019) for v in ZEBRAFISH_VIDEOS) : nothing
    for v in ZEBRAFISH_VIDEOS
        d = data[v]
        @printf("  %s: %d samples, Δ %.3f s, speed median %.2f / 95%% %.2f BL/s, |a| cut %.1f BL/s², neighbours within R: mean %.2f, %d coincident dropped%s\n",
            v, length(d.envs), d.Δ, d.speed..., d.amax, mean(length.(d.envs)), d.ncoincident, cross ? @sprintf(" (2019 tracking: %d samples)", length(data19[v].envs)) : "")
    end
    checkpoint("samples")
    vscale = data["zebrafish_80_1"].speed[2]   # the 95% speed quantile of a reference video, as a fixed scale
    kw = (cutoff = RCUT, vscale, inputmap = :tanh)
    mem = History(selfterm = 1:4, selfterm_degree = (8, 4), selfterm_maxm = 2)
    degrees = Dict("1" => (32,), "2" => (32, 12), "3" => (32, 12, 6))
    bases = [(prefix * n, dACEBasis(; maxdeg = degrees[n], maxl = 0, maxm = 1, history, kw...))
             for (prefix, history) in (("force", nothing), ("self", mem)) for n in ("1", "2", "3")]
    println("vscale $(round(vscale; digits = 2)) BL/s; basis sizes: ", join(["$o $(length(b))" for (o, b) in bases], ", "))
    stats = Dict((o, v) => design_statistics(b, data[v].envs, data[v].accs) for (o, b) in bases for v in ZEBRAFISH_VIDEOS)
    checkpoint("statistics")
    pred = Dict(o => SVector{2, Float64}[] for (o, _) in bases)
    pred19 = Dict(o => SVector{2, Float64}[] for (o, _) in bases)
    per_video = Dict{Tuple{String, String}, Float64}()
    per_video19 = Dict{Tuple{String, String}, Float64}()
    lams = Dict(o => NTuple{2, Float64}[] for (o, _) in bases)
    truth, truth19 = SVector{2, Float64}[], SVector{2, Float64}[]
    for v in ZEBRAFISH_VIDEOS
        rest = setdiff(ZEBRAFISH_VIDEOS, [v])
        for (o, b) in bases
            m = fit(b, reduce(+, [stats[(o, t)] for t in rest]); reflections = true)
            p = predict(m, data[v].envs)
            append!(pred[o], p)
            per_video[(o, v)] = R2(p, data[v].accs)
            push!(lams[o], (m.λ, m.λself))
            if cross
                p19 = predict(m, data19[v].envs)
                append!(pred19[o], p19)
                per_video19[(o, v)] = R2(p19, data19[v].accs)
            end
        end
        append!(truth, data[v].accs)
        cross && append!(truth19, data19[v].accs)
        @printf("  held out %-16s R² %s\n", v, join([@sprintf("%s %.4f", o, per_video[(o, v)]) for (o, _) in bases], ", "))
        flush(stdout)
    end
    r2 = Dict(o => R2(pred[o], truth) for (o, _) in bases)
    r2_19 = cross ? Dict(o => R2(pred19[o], truth19) for (o, _) in bases) : nothing
    wins(a, b) = count(per_video[(a, v)] > per_video[(b, v)] for v in ZEBRAFISH_VIDEOS)
    out = (; R = RCUT, gap = GAP, videos = ZEBRAFISH_VIDEOS, samples = length(truth), r2, per_video, lams,
        r2_2019 = r2_19, per_video_2019 = cross ? per_video19 : nothing,
        social_gain = r2["self3"] - r2["self1"],
        wins = (two_over_one = wins("self2", "self1"), three_over_two = wins("self3", "self2"), three_over_one = wins("self3", "self1")))
    tag = "R$(Int(RCUT))_gap$GAP"
    serialize(joinpath(RES, "zebrafish_$tag.jls"), out)
    open(joinpath(RES, "zebrafish_$tag.md"), "w") do io
        println(io, "# Zebrafish, leakage-free protocol, R = $RCUT BL, gap $GAP frames\n")
        println(io, "Leave one video out ($(length(ZEBRAFISH_VIDEOS)) videos, $(length(truth)) samples), staged self-term fits, pooled held-out R².\n")
        println(io, "| model | basis | R² (2023 tracking) |", cross ? " R² (2019 tracking) |" : "", "\n|---|---|---|", cross ? "---|" : "")
        for (o, b) in bases
            @printf(io, "| %s | %d | %.4f |%s\n", o, length(b), r2[o], cross ? @sprintf(" %.4f |", r2_19[o]) : "")
        end
        @printf(io, "\nsocial gain (self3 - self1) %+.4f; wins 2>1 %d, 3>2 %d, 3>1 %d (of %d)\n", out.social_gain, out.wins..., length(ZEBRAFISH_VIDEOS))
        println(io, "\n## Per video\n\n| video | ", join(first.(bases), " | "), " |\n|---|", "---|"^length(bases))
        for v in ZEBRAFISH_VIDEOS
            println(io, "| $v | ", join([@sprintf("%.4f", per_video[(o, v)]) for (o, _) in bases], " | "), " |")
        end
    end
    open(joinpath(RES, "zebrafish_protocol.jsonl"), "a") do io
        println(io, "{\"R\": $RCUT, \"gap\": $GAP, \"samples\": $(length(truth)), ",
            join(["\"$o\": {\"r2\": $(r2[o])" * (cross ? ", \"r2_2019\": $(r2_19[o])" : "") * "}" for (o, _) in bases], ", "),
            ", \"social_gain\": $(out.social_gain), \"wins_3_over_2\": $(out.wins.three_over_two), \"wins_3_over_1\": $(out.wins.three_over_one)}")
    end
    @printf("R %.0f gap %d: social gain %+.4f, 3>2 wins %d/%d, 3>1 wins %d/%d\n", RCUT, GAP, out.social_gain,
        out.wins.three_over_two, length(ZEBRAFISH_VIDEOS), out.wins.three_over_one, length(ZEBRAFISH_VIDEOS))
    checkpoint("leave one video out")
end

get(ENV, "ZEBRAFISH_NO_MAIN", "") == "1" || main()
