# julia -t auto --project=scripts/paper scripts/synthetic/chiral.jl
#
# Chiral rules (Figure 6): the 2- and 4-body models with complex and with real coefficients on the chiral
# collectives of setup.jl, splits 1-3, relative force error on the test simulations.
# out: results/chiral.csv (appended, one line per fit pair)

include(joinpath(@__DIR__, "..", "paper", "setup.jl"))

const RES = joinpath(@__DIR__, "results")
mkpath(RES)
const OUTF = joinpath(RES, "chiral.csv")
isfile(OUTF) || open(io -> println(io, "collective,model,split,complex,real,fit_s"), OUTF, "w")

for r in 1:3
    trs, tes = split_seeds(r)
    for s in CHIRAL_SWARMS
        train, test = swarm_data(s.rule, trs), swarm_data(s.rule, tes)
        for m in MODELS
            m.key == "d3" && continue
            b = m.basis()
            t = @elapsed begin
                S = design_statistics(b, train...)
                err = [rmse(fit(b, S; prior = prior_of(b), reflections), test...; relative = true) for reflections in (false, true)]
            end
            line = join((s.key, m.key, r, err..., round(t; digits = 1)), ",")
            open(io -> println(io, line), OUTF, "a")
            println(line); flush(stdout)
        end
    end
    checkpoint("chiral split $r")
end
