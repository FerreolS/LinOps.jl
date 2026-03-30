using Random
using LinearAlgebra: dot, mul!
using FFTW
using LinOps: LinOpDFT, LinOp, CoordinateSpace, inputspace, outputspace, inputsize, outputsize, isendomorphism

@testset "LinOpDFT - Real to complex 1D" begin
    N = 16
    F = LinOpDFT(Float64, (N,))

    @test F isa LinOp
    @test inputsize(F) == (N,)
    @test outputsize(F) == (div(N, 2) + 1,)
    @test !isendomorphism(F)

    x = randn(N)
    y = F * x
    @test size(y) == (div(N, 2) + 1,)
    @test y ≈ rfft(x)

    yc = randn(ComplexF64, div(N, 2) + 1)
    r = F' * yc
    @test size(r) == (N,)
    @test r ≈ brfft(yc, N)

    # adjoint identity does NOT hold for rfft/brfft (half-spectrum issue)
end

@testset "LinOpDFT - Real to complex 2D" begin
    M, N = 8, 12
    F = LinOpDFT(Float64, (M, N))

    @test inputsize(F) == (M, N)
    @test outputsize(F) == (div(M, 2) + 1, N)

    x = randn(M, N)
    @test F * x ≈ rfft(x)

    yc = randn(ComplexF64, div(M, 2) + 1, N)
    @test F' * yc ≈ brfft(yc, M)
end

@testset "LinOpDFT - Real to complex Float32" begin
    N = 16
    F = LinOpDFT(Float32, (N,))

    @test inputsize(F) == (N,)
    @test outputsize(F) == (div(N, 2) + 1,)

    x = randn(Float32, N)
    @test F * x ≈ rfft(x)
end

@testset "LinOpDFT - Complex to complex 1D" begin
    Random.seed!(42)
    N = 16
    F = LinOpDFT(ComplexF64, (N,))

    @test F isa LinOp
    @test inputsize(F) == (N,)
    @test outputsize(F) == (N,)
    @test isendomorphism(F)

    x = randn(ComplexF64, N)
    @test F * x ≈ fft(x)

    yv = randn(ComplexF64, N)
    @test F' * yv ≈ bfft(yv)

    # Adjoint identity: <F*x, y> == <x, F'*y>
    x = randn(ComplexF64, N)
    y = randn(ComplexF64, N)
    @test dot(F * x, y) ≈ dot(x, F' * y)

    # F * F' = N * I
    @test F * F' == N * I
    @test F' * F == N * I

    # in-place via mul!
    x = randn(ComplexF64, N)
    out = similar(x)
    mul!(out, F, x)
    @test out ≈ fft(x)

    out2 = similar(x)
    mul!(out2, F', out)
    @test out2 ≈ bfft(out)
end

@testset "LinOpDFT - Complex to complex 2D" begin
    Random.seed!(7)
    M, N = 8, 12
    F = LinOpDFT(ComplexF64, (M, N))

    @test inputsize(F) == (M, N)
    @test outputsize(F) == (M, N)

    x = randn(ComplexF64, M, N)
    @test F * x ≈ fft(x)

    # Adjoint identity
    x = randn(ComplexF64, M, N)
    y = randn(ComplexF64, M, N)
    @test dot(F * x, y) ≈ dot(x, F' * y)
end

@testset "LinOpDFT - Complex to complex ComplexF32" begin
    N = 16
    F = LinOpDFT(ComplexF32, (N,))

    @test inputsize(F) == (N,)
    @test outputsize(F) == (N,)

    x = randn(ComplexF32, N)
    @test F * x ≈ fft(x)
end

@testset "LinOpDFT - Summary and show" begin
    Fc = LinOpDFT(ComplexF64, (8, 16))
    Fr = LinOpDFT(Float64, (8, 16))

    sc = summary(Fc)
    sr = summary(Fr)
    @test occursin("LinOpDFT", sc)
    @test sprint(show, Fc) == sc
    @test occursin("LinOpDFT", sr)
    @test sprint(show, Fr) == sr
end
