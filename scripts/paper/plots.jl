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

# ---------------------------------------------------------------- Figure 2: the benchmark data

PLOTS["data"] = function (d)
    (; labels) = d
    ns = length(labels)
    # (a) one data point: neighbours within R as dots with velocity arrows, focal agent i, target a_i (orange)
    (; e_r, e_v, e_vj, a, R) = d
    fig = Figure(size = (0.3WIDTH, 0.3WIDTH), figure_padding = 2)
    ax = Axis(fig[1, 1]; aspect = AxisAspect(1), limits = (-1.1R, 1.1R, -1.1R, 1.1R))
    hidedecorations!(ax); hidespines!(ax)
    θs = range(0, 2π; length = 200)
    lines!(ax, R .* cos.(θs), R .* sin.(θs); color = INK2, linewidth = 0.75, linestyle = :dash)
    vlen = 0.25R / maximum(norm, [[e_v]; e_vj])
    for (r, vj) in zip(e_r, e_vj)
        lines!(ax, [Point2f(r), Point2f(r + vlen * vj)]; color = INK2, linewidth = 0.8)
        scatter!(ax, [Point2f(r + vlen * vj)]; marker = :rtriangle, rotation = atan(vj[2], vj[1]), markersize = 5, color = INK2)
        scatter!(ax, [Point2f(r)]; marker = :circle, markersize = 4.5, color = INK2, strokewidth = 0)
    end
    lines!(ax, [Point2f(0, 0), Point2f(vlen * e_v)]; color = INK, linewidth = 1.0)
    scatter!(ax, [Point2f(vlen * e_v)]; marker = :rtriangle, rotation = atan(e_v[2], e_v[1]), markersize = 6, color = INK)
    â = 0.45R * a / max(norm(a), 1.0e-12)
    lines!(ax, [Point2f(0, 0), Point2f(â)]; color = FORCE, linewidth = 1.6)
    scatter!(ax, [Point2f(â)]; marker = :rtriangle, rotation = atan(â[2], â[1]), markersize = 8, color = FORCE)
    scatter!(ax, [Point2f(0, 0)]; marker = :circle, markersize = 7, color = INK, strokewidth = 0)
    text!(ax, 0.0, 0.0; text = "i", offset = Tuple(-9 .* â ./ norm(â)), align = (:center, :center), fontsize = 8, color = INK)
    text!(ax, â[1], â[2]; text = rich("a", subscript("i")), offset = Tuple(10 .* â ./ norm(â)), align = (:center, :center), fontsize = 8, color = FORCE)
    text!(ax, R / sqrt(2), -R / sqrt(2); text = "R", align = (:left, :top), offset = (2, -2), fontsize = ANNOT, color = INK2)
    save(joinpath(OUT, "fig_data_point.pdf"), fig)
    # (b) neighbours per data point, 2D (grey) and 3D (black): median and quartiles over the training data
    (; n2, n3) = d
    fig = Figure(size = (0.66WIDTH, 0.3WIDTH), figure_padding = 2)
    ax = Axis(fig[1, 1]; xlabel = "number of neighbours within R", yticks = (1:ns, labels), yreversed = true,
        xlabelsize = LABEL, xticklabelsize = TICK, yticklabelsize = TICK)
    for k in 1:ns, (n, off, col, lab) in ((n2[k], -0.15, COLOR["grey"], "2D"), (n3[k], 0.15, INK, "3D"))
        q1, q2, q3 = quantile(n, (0.25, 0.5, 0.75))
        lines!(ax, [q1, q3], [k + off, k + off]; color = col, linewidth = 2.5)
        scatter!(ax, [q2], [k + off]; color = col, markersize = 5, label = k == 1 ? lab : nothing)
    end
    axislegend(ax; position = :rt, framevisible = false, labelsize = LEGEND, patchsize = (6, 6), rowgap = 0)
    save(joinpath(OUT, "fig_data_neighbours.pdf"), fig)
end

# ---------------------------------------------------------------- Figure 4: learned kernels

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
        ["true", "dACE 2-body (learned)", "dACE 3-body (learned)"]; patchsize = (14, 6))
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
