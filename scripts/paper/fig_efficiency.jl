# julia --project=scripts/paper scripts/paper/fig_efficiency.jl [outdir]
#
# Figure 8, from scripts/synthetic/efficiency.jl (results/efficiency.jsonl):
#   fig_efficiency_{pair,mixed,legend}.pdf  relative force error of dACE 4-body on clean test simulations against the
#     number of training samples, for the pairwise and 2+3+4-body rules, observation noise ε as the line style

isdefined(@__MODULE__, :SIM) || include(joinpath(@__DIR__, "setup.jl"))

let
    num(s, k) = parse(Float64, match(Regex("\"$k\"\\s*:\\s*([-+0-9.eE]+)"), s).captures[1])
    rows = [(swarm = match(r"\"swarm\"\s*:\s*\"([^\"]*)\"", l).captures[1], eps = num(l, "eps"), train = num(l, "train"), relerr = num(l, "relerr"))
        for l in eachline(joinpath(@__DIR__, "..", "synthetic", "results", "efficiency.jsonl"))]
    styles = ((0.0, :solid), (0.01, :dash), (0.1, :dot))
    for (k, swarm) in enumerate(("pair", "mixed"))
        fig = Figure(size = (0.49WIDTH, 165), figure_padding = 2)
        ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "training samples", ylabel = k == 1 ? "relative force error" : "",
            xticks = decades, xtickformat = DECADE_FORMAT, yticks = decades2, ytickformat = DECADE_FORMAT)
        for (ε, ls) in styles
            rs = sort([r for r in rows if r.swarm == swarm && r.eps == ε]; by = r -> r.train)
            isempty(rs) && continue
            scatterlines!(ax, [r.train for r in rs], [r.relerr for r in rs]; color = COLOR["d4"], linestyle = ls, markersize = 4, linewidth = 1)
        end
        save(joinpath(OUT, "fig_efficiency_$swarm.pdf"), fig)
    end
    legend_pdf("fig_efficiency_legend.pdf", [LineElement(color = COLOR["d4"], linestyle = ls) for (_, ls) in styles],
        ["dACE 4-body, ε = $(ε == 0 ? "0" : ε)" for (ε, _) in styles]; patchsize = (14, 6))
end
