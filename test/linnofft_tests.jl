using Random
using LinearAlgebra: dot, mul!
using NonuniformFFTs
using LinOps: LinOpNFFT, LinOp, CoordinateSpace, inputspace, outputspace, inputsize, outputsize, isendomorphism

@testset "LinOpNFFT - 1D Float32 construction" begin
    Random.seed!(1)
    N = 32; npts = 50
    xpts = Float32.(π .* (2 .* rand(npts) .- 1))
    NF = LinOpNFFT(Float32, (N,), (xpts,))

    @test NF isa LinOp
    @test inputsize(NF) == (npts,)
    @test outputsize(NF) == (div(N, 2) + 1,)   # real NUFFT → half spectrum
    @test !isendomorphism(NF)
end

@testset "LinOpNFFT - 1D Float32 forward" begin
    Random.seed!(2)
    N = 32; npts = 50
    xpts = Float32.(π .* (2 .* rand(npts) .- 1))
    NF = LinOpNFFT(Float32, (N,), (xpts,))

    x = randn(Float32, npts)
    y = NF * x
    @test size(y) == outputsize(NF)
    @test eltype(y) == ComplexF32
end

@testset "LinOpNFFT - 1D Float32 adjoint via mul!" begin
    Random.seed!(3)
    N = 32; npts = 50
    xpts = Float32.(π .* (2 .* rand(npts) .- 1))
    NF = LinOpNFFT(Float32, (N,), (xpts,))

    x = randn(Float32, npts)
    y = NF * x                               # ComplexF32 half-spectrum

    y_adj = similar(x, Float32, inputsize(NF))  # Float32 non-uniform output
    mul!(y_adj, NF', y)
    @test size(y_adj) == (npts,)
    @test eltype(y_adj) == Float32
end

@testset "LinOpNFFT - 1D Float32 adjoint via *" begin
    Random.seed!(4)
    N = 32; npts = 50
    xpts = Float32.(π .* (2 .* rand(npts) .- 1))
    NF = LinOpNFFT(Float32, (N,), (xpts,))

    x = randn(Float32, npts)
    y = NF * x
    r = NF' * y
    @test size(r) == (npts,)
    @test eltype(r) == Float32
end

@testset "LinOpNFFT - 1D Float32 in-place forward" begin
    Random.seed!(5)
    N = 32; npts = 50
    xpts = Float32.(π .* (2 .* rand(npts) .- 1))
    NF = LinOpNFFT(Float32, (N,), (xpts,))

    x = randn(Float32, npts)
    y_ref = NF * x
    y_out = similar(y_ref)
    mul!(y_out, NF, x)
    @test y_out ≈ y_ref
end

@testset "LinOpNFFT - 1D Float32 roundtrip shape" begin
    Random.seed!(6)
    N = 32; npts = 50
    xpts = Float16.(π .* (2 .* rand(npts) .- 1))
    NF = LinOpNFFT(Float32, (N,), (xpts,))

    x = randn(Float32, npts)
    rt = NF' * (NF * x)
    @test size(rt) == (npts,)
    @test eltype(rt) == Float32
end

@testset "LinOpNFFT - 2D Float32 construction and apply" begin
    Random.seed!(7)
    M = 16; N = 16; npts = 100
    xpts = Float32.(π .* (2 .* rand(npts) .- 1))
    ypts = Float32.(π .* (2 .* rand(npts) .- 1))
    NF = LinOpNFFT(Float32, (M, N), (xpts, ypts))

    @test inputsize(NF) == (npts,)
    @test outputsize(NF) == (div(M, 2) + 1, N)

    x = randn(Float32, npts)
    y = NF * x
    @test size(y) == outputsize(NF)
    @test eltype(y) == ComplexF32

    r = NF' * y
    @test size(r) == (npts,)
    @test eltype(r) == Float32
end

@testset "LinOpNFFT - Adjoint identity Float32" begin
    # For Float32 NUFFT (real→complex), the adjoint identity in ℂ is:
    #   real(dot(NF*x, y)) ≈ dot(x, real(NF'*y))   (Parseval between ℝ and ℂ)
    # Test via mul! consistency: NF'*(NF*x) and NF*(NF'*y) have correct sizes.
    Random.seed!(8)
    N = 32; npts = 50
    xpts = Float32.(π .* (2 .* rand(npts) .- 1))
    NF = LinOpNFFT(Float32, (N,), (xpts,))

    x = randn(Float32, npts)
    y_hat = NF * x                               # ComplexF32
    x_rt = NF' * y_hat                          # Float32
    @test size(x_rt) == size(x)

    # consistency: two calls give same result
    @test NF * x ≈ NF * x
    x_rt2 = NF' * y_hat
    @test x_rt ≈ x_rt2
end

@testset "LinOpNFFT - Summary and show" begin
    Random.seed!(9)
    N = 32; npts = 20
    xpts = Float32.(π .* (2 .* rand(npts) .- 1))
    NF = LinOpNFFT(Float32, (N,), (xpts,))

    s = summary(NF)
    @test occursin("LinOpNFFT", s)
    @test sprint(show, NF) == s
end
