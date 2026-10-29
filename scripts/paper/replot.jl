# julia --project=scripts/paper scripts/paper/replot.jl [outdir]
#
# Redraws every paper figure without refitting or resimulating: the figures of figures.jl and fig_data.jl from their
# cached plot data (OUT/plotdata, written by `plotfig`), the schematic, and the figures drawn from result files
# (recovery, chiral, efficiency, self-propulsion, tracked fish). Style changes (colours, text sizes, layout) then
# need only this script.

include(joinpath(@__DIR__, "setup.jl"))
replot_all()
for f in ("fig_schematic", "fig_recovery", "fig_chiral", "fig_efficiency", "fig_selfpropulsion", "fig_realdata")
    include(joinpath(@__DIR__, "$f.jl"))
    println("  drew $f")
end
