# julia --project=scripts/paper scripts/paper/fig_realdata.jl [outdir]
#
# Figures 10 and 11, from scripts/realdata/consistency.jl and shiners_protocol.jl (results/):
#   fig_realdata_{zebrafish,shiners}.pdf  disagreement between the force laws learned from two disjoint halves of
#     the data against the samples per half, single held-out groups faint and their mean bold
#   fig_shiner_saturation.pdf  the mean components of the learned 2-body part, the 3-body part and their sum along
#     the 2-body response, against the number of neighbours within 3 BL

isdefined(@__MODULE__, :SIM) || include(joinpath(@__DIR__, "setup.jl"))

const REAL = joinpath(@__DIR__, "..", "realdata", "results")
jsonnum(s, k) = parse(Float64, match(Regex("\"$k\"\\s*:\\s*([-+0-9.eE]+)"), s).captures[1])

# ---------------------------------------------------------------- consistency

let
    for (k, data) in enumerate(("zebrafish", "shiners"))
        recs = readlines(joinpath(REAL, "consistency_$data.jsonl"))
        fig = Figure(size = (0.5WIDTH, 165), figure_padding = 2)
        ax = Axis(fig[1, 1]; xlabel = "training samples per half", ylabel = k == 1 ? "disagreement between halves" : "", xscale = log10,
            xticks = ([1.0e3, 1.0e4, 1.0e5], ["10³", "10⁴", "10⁵"]), xlabelsize = LABEL, ylabelsize = LABEL, xticklabelsize = TICK, yticklabelsize = TICK)
        tests = unique(match(r"\"test\"\s*:\s*\"([^\"]*)\"", r).captures[1] for r in recs)
        curves = [sort([Point2f(jsonnum(r, "n"), jsonnum(r, "consistency")) for r in recs if occursin("\"test\": \"$t\"", r)]) for t in tests]
        for pts in curves   # each held-out group, faint
            lines!(ax, pts; color = (COLOR["d3"], 0.25), linewidth = 0.6)
        end
        # mean over the held-out groups at the sizes all of them reach; the largest size (all the data) differs
        # between groups and enters at its mean
        sizes = sort(collect(intersect((Set(round(Int, p[1]) for p in pts[1:(end - 1)]) for pts in curves)...)))
        mean_at(n) = mean(p[2] for pts in curves for p in pts if round(Int, p[1]) == n)
        xs = [sizes; mean(pts[end][1] for pts in curves)]
        ys = [mean_at.(sizes); mean(pts[end][2] for pts in curves)]
        o = sortperm(xs)
        lines!(ax, xs[o], ys[o]; color = COLOR["d3"], linewidth = 1.6, label = "dACE 3-body")
        k == 1 && axislegend(ax; position = :rt, framevisible = false, labelsize = LEGEND, patchsize = (8, 5), rowgap = 0)
        save(joinpath(OUT, "fig_realdata_$data.pdf"), fig)
    end
end

# ---------------------------------------------------------------- shiner saturation

# mean over the samples in bins of 5 neighbours (bins with at least 200 samples) of f(sample), a sample being
# (|2-body|, |2+3-body|, 2-body along, 3-body along, 2-body across, 3-body across the heading)
function saturation_curve(sat, f; width = 5)
    ns = sort(collect(keys(sat)))
    xs, ys = Float64[], Float64[]
    for b in 0:width:maximum(ns)
        v = [f(x) for n in ns if b <= n < b + width for x in sat[n]]
        length(v) >= 200 || continue
        push!(xs, b + width / 2); push!(ys, mean(v))
    end
    return xs, ys
end
# component of the 3-body part along the 2-body response (negative where it opposes it)
along2(x) = x[1] > 0 ? (x[3] * x[4] + x[5] * x[6]) / x[1] : 0.0

let
    sat = deserialize(joinpath(REAL, "shiners_R3_gap0.jls")).sat
    fig = Figure(size = (WIDTH, 200))
    ax = Axis(fig[1, 1]; xlabel = "neighbours within 3 BL", ylabel = "along the 2-body response (BL s⁻²)")
    hlines!(ax, [0]; color = INK2, linewidth = 0.75)
    lines!(ax, saturation_curve(sat, x -> x[1])...; color = COLOR["d2"], linestyle = :dash, label = "2-body part")
    lines!(ax, saturation_curve(sat, along2)...; color = COLOR["d3"], linestyle = :dash, label = "3-body part")
    lines!(ax, saturation_curve(sat, x -> x[1] + along2(x))...; color = INK2, label = "2- + 3-body")
    Legend(fig[1, 2], ax; patchsize = (14, 5), rowgap = 0, halign = :left)
    save(joinpath(OUT, "fig_shiner_saturation.pdf"), fig)
end
