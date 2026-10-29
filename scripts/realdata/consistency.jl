# CONSISTENCY_DATA=zebrafish|shiners julia -t 5 --project=scripts/paper scripts/realdata/consistency.jl
#
# Consistency of the learned force law on tracked fish (Section 4, Figure 10): for each held-out group, the
# memoryless 3-body law fitted on the first n samples of two disjoint halves of the training groups (alternating
# 20 s blocks, no shared frame), compared on the held-out group. CONSISTENCY_MATCHED=0: each half its own λ.
# out: results/consistency_<data>[_perhalf].{jsonl,md}

const DATASET = get(ENV, "CONSISTENCY_DATA", "zebrafish")
DATASET in ("zebrafish", "shiners") || error("CONSISTENCY_DATA must be zebrafish or shiners")
include(joinpath(@__DIR__, DATASET == "zebrafish" ? "zebrafish.jl" : "shiners.jl"))
using Random

const MATCHED = get(ENV, "CONSISTENCY_MATCHED", "1") == "1"
const RES = joinpath(@__DIR__, "results")
mkpath(RES)
const SIZES = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 10^9]   # per half; the last: all of it
const NPROBE = 5000
const BLOCK_S = 20.0   # time blocks alternate between the halves
const KFRAMES = 3      # the finite-difference step of both data sets
const VAL_EVERY = 8    # every 8th block of a half is held back from training
const GROUPS = DATASET == "zebrafish" ? ZEBRAFISH_VIDEOS : SHINER_TRIALS
const TESTS = DATASET == "zebrafish" ? ["zebrafish_60_1", "zebrafish_80_2", "zebrafish_100_3", "zebrafish_80_1"] :
    ["0105", "0115", "0107", "150-fish"]   # 10, 30, 70 and 150 fish
const RCUT = DATASET == "zebrafish" ? 8.0 : 3.0

# the time block of a sample, or -1 if its frames (inputs from t - K, target to t + 1 + 2K) reach across a block
# boundary: those samples are dropped, so that the two halves share no frame
function sampleblock(t, dt)
    lo, hi = floor(Int, (t - KFRAMES) * dt / BLOCK_S), floor(Int, (t + 1 + 2KFRAMES) * dt / BLOCK_S)
    return lo == hi ? lo : -1
end

# per group: environments, accelerations, the time block of each sample and the speed scale of the basis
function group_data(g)
    if DATASET == "zebrafish"
        d = zebrafish_samples(g; R = RCUT)
        _, dt, _, _ = load_zebrafish(g)
        return (; d.envs, d.accs, block = [sampleblock(t, dt) for (t, _) in d.keys], vscale = d.speed[2])
    end
    envs, accs, keys = shiner_samples(g; R = RCUT, keys = true)
    block = [(b = sampleblock(t, SHINER_DT); b < 0 ? b : 1000s + b) for (s, t, _) in keys]   # blocks numbered per segment
    return (; envs, accs, block, vscale = 3.0)
end

relrms(A, B) = sqrt(mean(norm.(A .- B) .^ 2) / ((mean(norm.(A) .^ 2) + mean(norm.(B) .^ 2)) / 2))
rms(A) = sqrt(mean(norm.(A) .^ 2))

function main()
    println("consistency on $DATASET: tests $TESTS, sizes per half $SIZES, λ $(MATCHED ? "matched" : "per half"), $(Threads.nthreads()) threads")
    data = Dict(g => group_data(g) for g in GROUPS)
    for g in GROUPS
        @printf("  %s: %d samples, %d blocks, %d dropped at block boundaries\n", g, length(data[g].envs),
            length(unique(filter(>=(0), data[g].block))), count(<(0), data[g].block))
    end
    basis = DATASET == "zebrafish" ?
        dACEBasis(; maxdeg = (32, 12, 6), maxl = 0, maxm = 1, inputmap = :tanh, cutoff = RCUT, vscale = data["zebrafish_80_1"].vscale) :
        dACEBasis(; maxdeg = (4, 6, 4), maxl = 0, maxm = 1, inputmap = :tanh, cutoff = RCUT, vscale = 3.0)
    γ = smoothness_prior(basis)
    println("basis: $(length(basis)) functions")
    rows = NamedTuple[]
    tag = "consistency_" * DATASET * (MATCHED ? "" : "_perhalf")
    out_jsonl = joinpath(RES, "$tag.jsonl")
    isfile(out_jsonl) && rm(out_jsonl)
    for test in TESTS
        rng = Xoshiro(hash(test))
        probe = sort(randperm(rng, length(data[test].envs))[1:min(NPROBE, length(data[test].envs))])
        penvs = data[test].envs[probe]
        # halves: alternating time blocks of each training group, in a fixed random order of samples
        pool = Dict(h => Tuple{String, Int}[] for h in (:A, :B))
        for g in setdiff(GROUPS, [test])
            bl = data[g].block
            rank = Dict(b => k for (k, b) in enumerate(sort(unique(filter(>=(0), bl)))))
            for (q, b) in enumerate(bl)
                b < 0 && continue   # frames in two blocks
                r = rank[b]
                (r ÷ 2) % VAL_EVERY == 0 && continue
                push!(pool[isodd(r) ? :A : :B], (g, q))
            end
        end
        order = Dict(h => shuffle(rng, pool[h]) for h in (:A, :B))
        nmax = min(length(order[:A]), length(order[:B]))   # the same sizes in both halves
        S = Dict{Symbol, Any}(:A => nothing, :B => nothing)
        ntr = Dict(:A => 0, :B => 0)
        add(S0, sel) = isempty(sel) ? S0 :
            (x = design_statistics(basis, [data[g].envs[q] for (g, q) in sel], [data[g].accs[q] for (g, q) in sel]); isnothing(S0) ? x : S0 + x)
        for n in unique(min.(SIZES, nmax))
            for h in (:A, :B)
                S[h] = add(S[h], order[h][(ntr[h] + 1):n]); ntr[h] = n
            end
            λs = if MATCHED
                λm = dACE.shared_evidence_lambda([S[h] for h in (:A, :B)]; prior = γ, reflections = true)
                Dict(:A => λm, :B => λm)
            else
                Dict(:A => :evidence, :B => :evidence)
            end
            fits = Dict(h => fit(basis, S[h]; λ = λs[h], prior = γ, reflections = true) for h in (:A, :B))
            F = Dict(h => predict(fits[h], penvs) for h in (:A, :B))
            soc = Dict(h => [sum(body_order_contributions(fits[h], e)[2:end]) for e in penvs] for h in (:A, :B))
            edge(λ) = λ in (first(dACE.EVIDENCE_GRID), last(dACE.EVIDENCE_GRID))
            row = (; data = DATASET, test, n, consistency = relrms(F[:A], F[:B]), consistency_social = relrms(soc[:A], soc[:B]),
                rms_A = rms(F[:A]), rms_B = rms(F[:B]), rms_social_A = rms(soc[:A]), rms_social_B = rms(soc[:B]),
                λ_A = fits[:A].λ, λ_B = fits[:B].λ, edge_A = edge(fits[:A].λ), edge_B = edge(fits[:B].λ))
            push!(rows, row)
            open(out_jsonl, "a") do io
                println(io, "{", join(["\"$k\": " * (v isa AbstractString ? "\"$v\"" : v isa Bool ? string(v) : @sprintf("%.6g", v)) for (k, v) in pairs(row)], ", "), "}")
            end
            @printf("  %s n %7d: consistency %.3f, λ %.1e / %.1e%s\n", test, n, row.consistency, row.λ_A, row.λ_B, row.edge_A || row.edge_B ? " (edge of the grid)" : "")
            flush(stdout)
        end
        checkpoint("consistency $test")
    end
    open(joinpath(RES, "$tag.md"), "w") do io
        println(io, "# Consistency of the learned force law, $DATASET (dACE 3-body, memoryless, λ $(MATCHED ? "matched between the halves" : "per half"))\n")
        println(io, "Two fits on disjoint halves (alternating $(Int(BLOCK_S)) s blocks) of the training groups, n samples each, compared on $NPROBE probe environments of the held-out group. consistency = RMS(F_A - F_B) / RMS(F_A, F_B), lower is better.\n")
        println(io, "| held out | n | consistency | social part | RMS F | RMS social | λ (A, B) |\n|---|---|---|---|---|---|---|")
        for r in rows
            @printf(io, "| %s | %d | %.3f | %.3f | %.3g | %.3g | %.0e%s, %.0e%s |\n", r.test, r.n, r.consistency, r.consistency_social,
                (r.rms_A + r.rms_B) / 2, (r.rms_social_A + r.rms_social_B) / 2, r.λ_A, r.edge_A ? "*" : "", r.λ_B, r.edge_B ? "*" : "")
        end
        println(io, "\n* λ at the edge of the grid.")
    end
    println("wrote ", joinpath(RES, "$tag.md"))
end

main()
