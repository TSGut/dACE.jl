# Drawing functions of the figures that need fits or simulations. Each takes the plot data that its computing
# script saved with `plotfig(name, data)` (OUT/plotdata/<name>.jls), so that replot.jl redraws every figure from
# the cache without refitting. Panel labels (a), (b), … are not drawn: multi-panel figures are written as one PDF
# per panel and labelled as subfigures in LaTeX.

using Serialization

const PLOTDATA = joinpath(OUT, "plotdata")
const PLOTS = Dict{String, Function}()

"Save the plot data of figure `name` and draw it."
function plotfig(name, data)
    mkpath(PLOTDATA)
    serialize(joinpath(PLOTDATA, "$name.jls"), data)
    return PLOTS[name](data)
end

"Redraw every figure whose plot data are cached."
function replot_all()
    for name in sort(collect(keys(PLOTS)))
        f = joinpath(PLOTDATA, "$name.jls")
        isfile(f) || (println("  no plot data for $name"); continue)
        PLOTS[name](deserialize(f))
        println("  drew $name")
    end
end

"A legend shared by several panels, as its own PDF placed under the subfigures."
function legend_pdf(file, elems, labels; width = WIDTH, nbanks = 1, patchsize = (8, 6))
    fig = Figure(size = (width, 11nbanks + 4), figure_padding = 1)
    Legend(fig[1, 1], elems, labels; orientation = :horizontal, nbanks, framevisible = false, labelsize = LEGEND, patchsize,
        colgap = 8, rowgap = 0, padding = (0, 0, 0, 0))
    return save(joinpath(OUT, file), fig)
end

# ---------------------------------------------------------------- Figure 3: learned kernels

const TRUEBAND = colorant"#cfccc4"   # true kernels: a wide pale band under the learned curves
PLOTS["kernels"] = function (d)
    (; ρs, true_rad, rad3, rad2, true_align, align3, align2, φs, flank) = d
    panel() = Figure(size = (0.32WIDTH, 125), figure_padding = 2)
    fig = panel()
    ax = Axis(fig[1, 1]; xlabel = "distance ρ", ylabel = "force along r̂")
    hlines!(ax, 0; color = INK2, linewidth = 0.75)
    lines!(ax, ρs, true_rad; color = TRUEBAND, linewidth = 6)
    lines!(ax, ρs, rad3; color = COLOR["d3"], linewidth = 1.5)
    lines!(ax, ρs, rad2; color = COLOR["d2"], linewidth = 1.5)
    save(joinpath(OUT, "fig_kernels_radial.pdf"), fig)
    fig = panel()
    ax = Axis(fig[1, 1]; xlabel = "distance ρ", ylabel = "response to uᵢⱼ")
    lines!(ax, ρs, true_align; color = TRUEBAND, linewidth = 6)
    lines!(ax, ρs, align3; color = COLOR["d3"], linewidth = 1.5)
    lines!(ax, ρs, align2; color = COLOR["d2"], linewidth = 1.5)
    save(joinpath(OUT, "fig_kernels_alignment.pdf"), fig)
    fig = panel()
    ax = Axis(fig[1, 1]; xlabel = "opening angle φ", ylabel = "response to uᵢⱼ = uᵢₖ", xticks = ([0, π / 2, π], ["0", "π/2", "π"]))
    for (f, ls, ρ) in zip(flank, (:solid, :dot), (1, 2))
        lines!(ax, φs, f.truth; color = TRUEBAND, linewidth = 6)
        lines!(ax, φs, f.learned; color = COLOR["d3"], linestyle = ls, linewidth = 1.5)
        k = round(Int, 0.7length(φs))
        text!(ax, φs[k], f.learned[k]; text = "ρ = $ρ", align = (:right, :bottom), offset = (-3, 3), fontsize = ANNOT, color = INK2)
    end
    save(joinpath(OUT, "fig_kernels_flanking.pdf"), fig)
    legend_pdf("fig_kernels_legend.pdf", [LineElement(color = TRUEBAND, linewidth = 6), LineElement(color = COLOR["d2"], linewidth = 1.5), LineElement(color = COLOR["d3"], linewidth = 1.5)],
        ["true", "dACE 2-body", "dACE 3-body"]; patchsize = (14, 6))
end

# ---------------------------------------------------------------- trajectory predictions

for name in ("rollout_mixed", "rollout_mixed3d")
    PLOTS[name] = function (d)
        ex = (; d.ex..., models = Tuple(k for k in d.ex.models if haskey(LINESTYLE, k)))   # the models of the paper
        d = (; d..., ex)
        project = isnothing(d.pca) ? identity : (A -> reshape(d.pca * reshape(A, size(d.pca, 2), :), 2, size(A, 2), size(A, 3)))
        rollout_figure(joinpath(OUT, "fig_$name.pdf"), d.ex; project, reference = d.reference)
    end
end
