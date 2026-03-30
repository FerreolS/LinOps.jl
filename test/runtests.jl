using LinOps
using LinOps: AbstractDomain, CoordinateSpace

using FFTW
using NonuniformFFTs

using Test
@testset "LinOps" begin
    include("domain_tests.jl")
    include("linop_tests.jl")
    include("linopdiag_tests.jl")
    include("lindft_tests.jl")
    include("linnofft_tests.jl")
end
