# julia --project=scripts/paper scripts/paper/fig_selfpropulsion.jl [outdir]
#
# Figure 9, from scripts/synthetic/stochastic.jl (results/stochastic.jsonl, stochastic_self.jsonl), one PDF per panel:
#   fig_selfpropulsion_past.pdf  next-step R² against the relative error of the learned force, the fit without past
#     joined to the one with the past acceleration as an input, per data set
#   fig_selfpropulsion_self.pdf  the relative error of the social part and R² of the staged and joint self-term fits,
#     each joined to the fit without self term

isdefined(@__MODULE__, :SIM) || include(joinpath(@__DIR__, "setup.jl"))

const STOCH = joinpath(@__DIR__, "..", "synthetic", "results")
jsonnum(s, k) = parse(Float64, match(Regex("\"$k\"\\s*:\\s*([-+0-9.eE]+)"), s).captures[1])
# the object of key `k` in a one-line JSON record (no nesting below it)
jsonobj(s, k) = "{" * match(Regex("\"$k\"\\s*:\\s*\\{([^{}]*)\\}"), s).captures[1] * "}"

let
    fig = Figure(size = (0.5WIDTH, 250), figure_padding = 2)
    ax = Axis(fig[1, 1]; xlabel = "next-step R²", ylabel = "relative error of the learned force",
        yscale = log10, yticks = ([0.4, 0.6, 1.0, 1.5], ["0.4", "0.6", "1", "1.5"]), xlabelsize = LABEL, ylabelsize = LABEL, xticklabelsize = TICK, yticklabelsize = TICK)
    for l in eachline(joinpath(STOCH, "stochastic.jsonl"))
        pt(k) = (o = jsonobj(l, k); Point2f(jsonnum(o, "onestep_R2"), jsonnum(o, "force_relerr")))
        p0, p1 = pt("iid"), pt("memory_onebody")
        lines!(ax, [p0, p1]; color = (COLOR["d4"], 0.4), linewidth = 0.6)
        scatter!(ax, [p0]; color = COLOR["d4"], marker = :circle, markersize = 5)
        scatter!(ax, [p1]; color = COLOR["d4"], marker = :utriangle, markersize = 6)
    end
    elems = [MarkerElement(color = COLOR["d4"], marker = :circle, markersize = 5), MarkerElement(color = COLOR["d4"], marker = :utriangle, markersize = 6)]
    Legend(fig[2, 1], elems, ["dACE 4-body, no past", "dACE 4-body, past acceleration"];
        orientation = :vertical, framevisible = false, labelsize = LEGEND, patchsize = (5, 5), rowgap = 0, tellheight = true, halign = :left)
    rowgap!(fig.layout, 2)
    save(joinpath(OUT, "fig_selfpropulsion_past.pdf"), fig)
end

let
    fig = Figure(size = (0.5WIDTH, 250), figure_padding = 2)
    ax = Axis(fig[1, 1]; xlabel = "next-step R²", ylabel = "relative error of the social part",
        xlabelsize = LABEL, ylabelsize = LABEL, xticklabelsize = TICK, yticklabelsize = TICK)
    for l in eachline(joinpath(STOCH, "stochastic_self.jsonl"))
        pt(k) = (o = jsonobj(l, k); Point2f(jsonnum(o, "onestep_R2"), jsonnum(o, "social_relerr")))
        p0, ps, pj = pt("iid"), pt("staged"), pt("self4")
        lines!(ax, [p0, ps]; color = (COLOR["d4"], 0.4), linewidth = 0.6)
        lines!(ax, [p0, pj]; color = (PURPLE, 0.4), linewidth = 0.6, linestyle = :dash)
        scatter!(ax, [p0]; color = COLOR["d4"], marker = :circle, markersize = 5)
        scatter!(ax, [ps]; color = COLOR["d4"], marker = :diamond, markersize = 6)
        scatter!(ax, [pj]; color = PURPLE, marker = :xcross, markersize = 6)
    end
    elems = [MarkerElement(color = COLOR["d4"], marker = :circle, markersize = 5), MarkerElement(color = COLOR["d4"], marker = :diamond, markersize = 6),
        MarkerElement(color = PURPLE, marker = :xcross, markersize = 6)]
    Legend(fig[2, 1], elems, ["dACE 4-body, no self term", "dACE 4-body, staged self term", "dACE 4-body, joint self term"];
        orientation = :vertical, framevisible = false, labelsize = LEGEND, patchsize = (5, 5), rowgap = 0, tellheight = true, halign = :left)
    rowgap!(fig.layout, 2)
    save(joinpath(OUT, "fig_selfpropulsion_self.pdf"), fig)
end
