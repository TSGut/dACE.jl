# SHINER_R=3 SHINER_GAP=0 julia -t 5 --project=scripts/paper scripts/realdata/shiners_protocol.jl
#
# Golden shiners (Section 4): leave one trial out, the 1-body law without past and the 1-, 2- and 3-body models
# with the self term, scored by held-out next-step R². At gap 0 also the 3-body saturation (Figure 11).
# out: results/shiners_R<R>_gap<G>.{jls,md}, a line in results/shiners_protocol.jsonl

include(joinpath(@__DIR__, "shiners.jl"))

const RCUT = parse(Float64, get(ENV, "SHINER_R", "3"))
const GAP = parse(Int, get(ENV, "SHINER_GAP", "0"))
const RES = joinpath(@__DIR__, "results")
mkpath(RES)
R2(p, a) = 1 - sum(norm.(p - a) .^ 2) / sum(norm.(a) .^ 2)

# the interaction part of a staged fit: its interaction coefficients, the self term zeroed
interaction_part(m) = (K = length(m.basis.funcs); dACEModel(m.basis, [m.c[1:K]; zeros(length(m.basis) - K)]))

# cancellation of the 2-body part by the 3-body part on environments with at least `nmin` neighbours:
# -Σ C2·C3 / Σ |C2|² and RMS(C2 + C3) / RMS(C2)
function cancellation(m, envs; nmin = 20)
    C = [body_order_contributions(m, x) for x in envs if length(x.r) >= nmin]
    isempty(C) && return (n = 0, cancellation = NaN, ratio = NaN)
    c2c3, c22 = sum(dot(c[2], c[3]) for c in C), sum(norm(c[2])^2 for c in C)
    return (n = length(C), cancellation = -c2c3 / c22, ratio = sqrt(sum(norm(c[2] + c[3])^2 for c in C) / c22))
end

function main()
    println("golden shiners, protocol: R = $RCUT BL, gap $GAP frames, $(Threads.nthreads()) threads")
    data = Dict(t => shiner_samples(t; R = RCUT, gap = GAP) for t in SHINER_TRIALS)
    foreach(t -> println("  $t: $(length(data[t][1])) samples"), SHINER_TRIALS)
    kw = (cutoff = RCUT, vscale = 3.0, inputmap = :tanh)
    mem = History(selfterm = 1:4, selfterm_degree = (8, 4), selfterm_maxm = 2)
    bases = [("force1", dACEBasis(; maxdeg = (4,), kw...)), ("self1", dACEBasis(; maxdeg = (4,), history = mem, kw...)),
        ("self2", dACEBasis(; maxdeg = (4, 6), maxl = 0, maxm = 1, history = mem, kw...)),
        ("self3", dACEBasis(; maxdeg = (4, 6, 4), maxl = 0, maxm = 1, history = mem, kw...))]
    println("basis sizes: ", join(["$o $(length(b)) ($(length(b.funcs)) interaction)" for (o, b) in bases], ", "))
    stats = Dict((o, t) => design_statistics(b, data[t]...) for (o, b) in bases for t in SHINER_TRIALS)
    fitstaged(o, b, train) = fit(b, reduce(+, [stats[(o, t)] for t in train]); reflections = true)
    pred = Dict(o => SVector{2, Float64}[] for (o, _) in bases)
    err = Dict(o => Float64[] for (o, _) in bases)
    lams = Dict(o => NTuple{2, Float64}[] for (o, _) in bases)
    truth = SVector{2, Float64}[]
    for t in SHINER_TRIALS
        rest = setdiff(SHINER_TRIALS, [t])
        for (o, b) in bases
            m = fitstaged(o, b, rest)
            p = predict(m, data[t][1])
            append!(pred[o], p)
            push!(err[o], 1 - R2(p, data[t][2]))
            push!(lams[o], (m.λ, m.λself))
        end
        append!(truth, data[t][2])
        @printf("  trial %-8s R² %s\n", t, join([@sprintf("%s %.4f", o, 1 - err[o][end]) for (o, _) in bases], ", "))
        flush(stdout)
    end
    r2 = Dict(o => R2(pred[o], truth) for (o, _) in bases)
    wins(a, b) = count(err[a] .< err[b])
    out = Dict{Symbol, Any}(:R => RCUT, :gap => GAP, :trials => SHINER_TRIALS, :samples => length(truth), :r2 => r2, :err => err, :lams => lams,
        :social_gain => r2["self3"] - r2["self1"],
        :wins => (two_over_one = wins("self2", "self1"), three_over_two = wins("self3", "self2"), three_over_one = wins("self3", "self1")))
    if GAP == 0
        # saturation of the interaction part (the memoryless interactions of the staged fit) fitted on all trials
        b3 = only(b for (o, b) in bases if o == "self3")
        mi3 = interaction_part(fitstaged("self3", b3, SHINER_TRIALS))
        sat = Dict{Int, Vector{NTuple{6, Float64}}}()
        for t in SHINER_TRIALS, x in data[t][1][1:4:end]
            C = body_order_contributions(mi3, x)
            h = x.v / norm(x.v)
            nrm = SVector(-h[2], h[1])
            push!(get!(sat, length(x.r), NTuple{6, Float64}[]), (norm(C[2]), norm(C[2] + C[3]), dot(C[2], h), dot(C[3], h), dot(C[2], nrm), dot(C[3], nrm)))
        end
        out[:sat] = sat
        # leave one trial out: the cancellation on the held-out trial of a fit on the other nine
        out[:cancellation] = [(; trial = t, cancellation(interaction_part(fitstaged("self3", b3, setdiff(SHINER_TRIALS, [t]))), data[t][1])...) for t in SHINER_TRIALS]
    end
    tag = "R$(Int(RCUT))_gap$GAP"
    serialize(joinpath(RES, "shiners_$tag.jls"), (; out...))
    open(joinpath(RES, "shiners_$tag.md"), "w") do io
        println(io, "# Golden shiners, leakage-free protocol, R = $RCUT BL, gap $GAP frames\n")
        println(io, "Leave one trial out ($(length(SHINER_TRIALS)) trials, $(length(truth)) samples), staged self-term fits, pooled held-out R².\n")
        println(io, "| model | basis | R² |\n|---|---|---|")
        for (o, b) in bases
            @printf(io, "| %s | %d | %.4f |\n", o, length(b), r2[o])
        end
        @printf(io, "\nsocial gain (self3 - self1) %+.4f; wins 2>1 %d, 3>2 %d, 3>1 %d (of %d)\n", out[:social_gain], out[:wins]..., length(SHINER_TRIALS))
        if GAP == 0
            println(io, "\n## 3-body cancellation on the held-out trial (≥ 20 neighbours)\n\n| held out | samples | -Σ C2·C3 / Σ |C2|² | RMS(C2+C3) / RMS(C2) |\n|---|---|---|---|")
            for r in out[:cancellation]
                @printf(io, "| %s | %d | %.3f | %.3f |\n", r.trial, r.n, r.cancellation, r.ratio)
            end
        end
    end
    open(joinpath(RES, "shiners_protocol.jsonl"), "a") do io
        println(io, "{\"R\": $RCUT, \"gap\": $GAP, \"samples\": $(length(truth)), ",
            join(["\"$o\": {\"r2\": $(r2[o])}" for (o, _) in bases], ", "),
            ", \"social_gain\": $(out[:social_gain]), \"wins_3_over_2\": $(out[:wins].three_over_two), \"wins_3_over_1\": $(out[:wins].three_over_one)}")
    end
    @printf("R %.0f gap %d: social gain %+.4f, 3>2 wins %d/%d\n", RCUT, GAP, out[:social_gain], out[:wins].three_over_two, length(SHINER_TRIALS))
end

get(ENV, "SHINER_NO_MAIN", "") == "1" || main()
