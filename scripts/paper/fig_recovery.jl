# julia --project=scripts/paper scripts/paper/fig_recovery.jl [outdir]
#
# Figure 3, from the split results of figures.jl (OUT/splits.jls):
#   fig_recovery_{2d,3d,legend}.pdf  relative force error of every collective and model in 2D and 3D, median and range

isdefined(@__MODULE__, :SIM) || include(joinpath(@__DIR__, "setup.jl"))

isfile(joinpath(OUT, "splits.jls")) || error("no split results in $OUT: run figures.jl first")
splits = deserialize(joinpath(OUT, "splits.jls")).splits   # 3D results on the first NSPLITS3 splits only (haskey below)

lo_med_hi(v) = (minimum(v), median(v), maximum(v))

# one PDF per panel (2D, 3D) and the shared legend
let
    models = [(m.key, m.label) for m in MODELS]   # the same keys and labels as MODELS3
    labels = [s.label for s in SWARMS]
    # common baseline of the bars on the log axis of both panels, below the smallest median
    allvals(field) = [getfield(sp, field)[(s.key, mk)] for sp in splits, s in SWARMS, (mk, _) in models if haskey(getfield(sp, field), (s.key, mk))]
    base = 10.0^floor(log10(0.5minimum([allvals(:errors); allvals(:errors3)])))
    for (file, field, ylabel) in (("fig_recovery_2d", :errors, "relative force error"), ("fig_recovery_3d", :errors3, ""))
        fig = Figure(size = (0.5WIDTH, 230), figure_padding = 2)
        nm = length(models)
        # one group of bars per collective, two empty slots between groups
        gapw = 2
        centre(k) = (k - 1) * (nm + gapw) + (nm + 1) / 2
        ax = Axis(fig[1, 1]; yscale = log10, ylabel, xticks = ([centre(k) for k in 1:length(SWARMS)], labels), xticklabelrotation = π / 5,
            yticks = decades2, ytickformat = DECADE_FORMAT, xticklabelsize = TICK, yticklabelsize = TICK, ylabelsize = LABEL, xgridvisible = false)
        for (j, (mk, _)) in enumerate(models), (k, s) in enumerate(SWARMS)
            v = [getfield(sp, field)[(s.key, mk)] for sp in splits if haskey(getfield(sp, field), (s.key, mk))]
            isempty(v) && continue
            a, m, b = lo_med_hi(v)
            x = (k - 1) * (nm + gapw) + j
            barplot!(ax, [x], [m]; fillto = base, color = COLOR[mk], width = 1.0, gap = 0, strokewidth = 0)
            rangebars!(ax, [x], [a], [b]; color = INK, linewidth = 0.6, whiskerwidth = 2)
        end
        ylims!(ax, base, 2.0)
        save(joinpath(OUT, "$file.pdf"), fig)
    end
    legend_pdf("fig_recovery_legend.pdf", [PolyElement(color = COLOR[k]) for (k, _) in models], [l for (_, l) in models])
end
