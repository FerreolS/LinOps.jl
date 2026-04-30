using LinOps
using LinOps: AbstractDomain, CoordinateSpace

using FFTW
using NonuniformFFTs

using Test
@testset "LinOps" begin
    include("domain_tests.jl")
    include("linop_tests.jl")
    include("operations_tests.jl")
    include("linopmapslice_tests.jl")
    include("linopdiag_tests.jl")
    include("linopgrad_tests.jl")
    include("lindft_tests.jl")
    include("linnofft_tests.jl")

    # GPU tests only if GPUArrays and JLArrays are available
    try
        using GPUArrays, JLArrays
        include("linopgpu_tests.jl")
    catch
        @warn "Skipping GPU tests (GPUArrays/JLArrays not available)"
    end

    # Zygote autodiff tests
    try
        using Zygote, FiniteDifferences
        include("linopautodiff_tests.jl")
    catch e
        @warn "Skipping Zygote autodiff tests" exception = e
    end
end
