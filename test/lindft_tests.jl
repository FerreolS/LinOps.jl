using Random
using LinearAlgebra: dot, mul!
using Adapt
using FFTW
using LinOps: LinOpDFT, LinOpDiag, LinOp, CoordinateSpace, inputspace, outputspace, inputsize, outputsize, inputtype, outputtype, isendomorphism

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

@testset "LinOpDFT - Generic composition and fallback mul!" begin
    N = 16
    F = LinOpDFT(ComplexF64, (N,))
    G = LinOpDFT(ComplexF64, (N,))
    x = randn(ComplexF64, N)

    C = F * G
    @test C isa LinOp
    @test C * x ≈ fft(fft(x))

    y = similar(x)
    mul!(y, C, x)
    @test y ≈ fft(fft(x))

    @test F != LinOpDiag(ones(N))

    H = LinOpDFT(ComplexF64, (N + 2,))
    @test_throws ArgumentError F * H
end

@testset "LinOpDFT - Distinct adjoint compositions" begin
    N = 16
    F = LinOpDFT(ComplexF64, (N,))
    G = LinOpDFT(ComplexF64, (N,))
    x = randn(ComplexF64, N)

    C1 = F * G'
    @test C1 isa LinOp
    @test C1 * x ≈ fft(bfft(x))

    C2 = F' * G
    @test C2 isa LinOp
    @test C2 * x ≈ bfft(fft(x))
end

@testset "LinOpDFT - Adjoint and inverse" begin
    N = 16
    F = LinOpDFT(ComplexF64, (N,))
    @test inv(F) * 4 * F isa UniformScaling
    @test inv(F) * 4 * F == UniformScaling(4)
    @test @inferred inv(F).left == UniformScaling(1 / N)
    x = randn(ComplexF64, N)

    y = F * x
    @test @inferred inv(F) * y ≈ x

end

@testset "LinOpDFT - Adjoint of generic composition" begin
    N = 16
    F = LinOpDFT(ComplexF64, (N,))
    C = F * F
    x = randn(ComplexF64, N)

    @test C' * x ≈ bfft(bfft(x))

    y = similar(x)
    mul!(y, C', x)
    @test y ≈ bfft(bfft(x))
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

@testset "LinOpDFT - FFTW extension adapt_structure" begin
    Fr = LinOpDFT(Float64, (8, 6))
    Ar = Adapt.adapt(Array{Float32}, Fr)
    @test Ar isa LinOp
    @test inputsize(Ar) == inputsize(Fr)
    @test inputtype(Ar) == Float32

    Fc = LinOpDFT(ComplexF64, (8, 6))
    Ac = Adapt.adapt(Array{ComplexF32}, Fc)
    @test Ac isa LinOp
    @test inputsize(Ac) == inputsize(Fc)
    @test outputsize(Ac) == outputsize(Fc)
    @test outputtype(Ac, rand(ComplexF32, inputsize(Ac)...)) == ComplexF32
end

@testset "LinOpDFT - FFTW extension flag validation" begin
    badflags = typemax(Int)
    @test_throws "only FFTW planning flags can be specified" LinOpDFT(Float64, (8,); flags = badflags)
end
