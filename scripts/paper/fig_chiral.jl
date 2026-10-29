# julia --project=scripts/paper scripts/paper/fig_chiral.jl [outdir]
#
# Figure 7, from the fits of the chiral collectives with complex and with real coefficients
# (scripts/synthetic/chiral.jl, results/chiral.csv):
#   fig_chiral{,_legend}.pdf  relative force error of dACE 4-body, complex and real, on every chiral collective, median and range

isdefined(@__MODULE__, :SIM) || include(joinpath(@__DIR__, "setup.jl"))

let
    rows = split.(readlines(joinpath(@__DIR__, "..", "synthetic", "results", "chiral.csv"))[2:end], ',')
    dace(key, col) = [parse(Float64, r[col]) for r in rows if r[1] == key && r[2] == "d4"]
    series = [("dACE 4-body, complex", COLOR["d4"], key -> dace(key, 4)), ("dACE 4-body, real", colorant"#a9d6a0", key -> dace(key, 5))]
    labels = [replace(s.label, ", " => "\n") for s in CHIRAL_SWARMS]
    nm, gapw = length(series), 2
    centre(k) = (k - 1) * (nm + gapw) + (nm + 1) / 2
    base = 1.0e-9
    fig = Figure(size = (0.6WIDTH, 170), figure_padding = 2)
    ax = Axis(fig[1, 1]; yscale = log10, ylabel = "relative force error", xticks = ([centre(k) for k in eachindex(labels)], labels),
        yticks = decades2, ytickformat = DECADE_FORMAT, xticklabelsize = TICK, yticklabelsize = TICK, ylabelsize = LABEL, xgridvisible = false)
    for (j, (_, colour, values)) in enumerate(series), (k, s) in enumerate(CHIRAL_SWARMS)
        v = values(s.key)
        isempty(v) && continue
        x = (k - 1) * (nm + gapw) + j
        barplot!(ax, [x], [median(v)]; fillto = base, color = colour, width = 1.0, gap = 0, strokewidth = 0)
        rangebars!(ax, [x], [minimum(v)], [maximum(v)]; color = INK, linewidth = 0.6, whiskerwidth = 2)
    end
    ylims!(ax, base, 2.0)
    save(joinpath(OUT, "fig_chiral.pdf"), fig)
    legend_pdf("fig_chiral_legend.pdf", [PolyElement(color = c) for (_, c, _) in series], [l for (l, _, _) in series]; width = 0.6WIDTH)
end
