# julia --project=scripts/paper scripts/paper/replot.jl [outdir]
#
# Redraws every paper figure from cached plot data and result files and rewrites numbers.tex from the saved split
# results, without refitting or resimulating.

include(joinpath(@__DIR__, "setup.jl"))
replot_all()
for f in ("fig_schematic", "fig_recovery", "fig_chiral", "fig_efficiency", "fig_selfpropulsion", "fig_realdata")
    include(joinpath(@__DIR__, "$f.jl"))
    println("  drew $f")
end
let f = joinpath(OUT, "splits.jls")
    isfile(f) ? (r = deserialize(f); write_numbers(r.nsplits3, r.splits); println("  wrote numbers.tex")) :
        println("  no split results in $OUT, numbers.tex not written")
end
