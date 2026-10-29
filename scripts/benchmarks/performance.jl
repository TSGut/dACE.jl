# Run from the repository root: julia --project=. -t 4 scripts/benchmarks/performance.jl
# Warmed medians and minimum allocated bytes; no timing assertions in the test suite.
using dACE, LinearAlgebra, Random, StaticArrays
BLAS.set_num_threads(1)

function measure(label, f; samples = 9)
    f(); f()
    results = [@timed f() for _ in 1:samples]
    times = sort!([r.time for r in results])
    println(rpad(label, 32), " ", round(1000times[cld(samples, 2)]; digits = 4),
        " ms; ", minimum(r.bytes for r in results), " bytes")
end

function benchmark()
    rng = Xoshiro(42)
    println("Julia ", VERSION, "; Julia threads: ", Threads.nthreads(), "; BLAS threads: ", BLAS.get_num_threads())
    env = Environment(SVector(1.0, 0.2),
        [normalize(randn(rng, SVector{2})) * (0.3 + 4rand(rng)) for _ in 1:30],
        [randn(rng, SVector{2}) for _ in 1:30])
    b = dACEBasis(maxdeg = (4, 8, 6), maxl = 0, maxm = 1)
    work = EvaluationWorkspace(b)
    out = zeros(ComplexF64, length(b))
    m = dACEModel(b, randn(rng, ComplexF64, length(b)))
    envs = fill(env, 256)
    measure("construct 2D", () -> dACEBasis(maxdeg = (4, 8, 6), maxl = 0, maxm = 1))
    measure("evaluate 2D", () -> evaluate(b, env))
    measure("evaluate! 2D", () -> evaluate!(out, b, env, work))
    measure("predict 2D", () -> predict(m, env))
    measure("predict 256 environments", () -> predict(m, envs))
    measure("design 256 environments", () -> dACE.design_matrix(b, envs))
    B = dACE.design_matrix(b, envs)
    G = B' * B
    h = B' * ones(ComplexF64, length(envs))
    measure("normal-equation solve", () -> dACE.solve_normal_equations(G, h; λ = 0.01))
    measure("QR statistics 256 environments", () -> dACE.qr_statistics(b, envs, ones(ComplexF64, length(envs))))
    S = dACE.qr_statistics(b, envs, ones(ComplexF64, length(envs)))
    measure("streamed-QR solve", () -> dACE.lsq_solve(S; λ = 0.01))
    b3 = FrameBasis(maxdeg = (4, 4, 3), maxm = 1, maxl = 0, maxa = 1, maxb = 1, vertical = true)
    e3 = Environment3(SVector(1.0, 0.2, 0.3),
        [normalize(randn(rng, SVector{3})) * (0.3 + 4rand(rng)) for _ in 1:30],
        [randn(rng, SVector{3}) for _ in 1:30])
    w3 = EvaluationWorkspace(b3)
    out3 = zeros(ComplexF64, length(b3))
    measure("evaluate 3D", () -> evaluate(b3, e3))
    measure("evaluate! 3D", () -> evaluate!(out3, b3, e3, w3))
    for N in (100, 1000, 5000)
        X, V = 3sqrt(N) * rand(rng, 2, N), randn(rng, 2, N)
        measure("environments N=$N", () -> environments(X, V; cutoff = 5.0))
    end
end

benchmark()
