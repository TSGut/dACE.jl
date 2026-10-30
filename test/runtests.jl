using dACE, dACESwarms, Test, LinearAlgebra, Random, Statistics, StaticArrays

include("helpers.jl")

@testset "dACE" begin
    include("basis.jl")
    include("history.jl")
    include("fitting.jl")
    include("swarms.jl")
    include("validation.jl")
    include("fastpaths.jl")
end
