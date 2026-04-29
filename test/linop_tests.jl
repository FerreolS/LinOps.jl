using LinearAlgebra: I, UniformScaling, mul!
using LinOps: LinOp, LinOpDiag, CoordinateSpace, inputsize, outputsize, inputspace, outputspace
import LinOps: apply_, apply_adjoint_

struct ApplyOnlyOp <: LinOp{CoordinateSpace{1}, CoordinateSpace{1}}
    inputspace::CoordinateSpace{1}
    outputspace::CoordinateSpace{1}
end

ApplyOnlyOp(n::Int) = ApplyOnlyOp(CoordinateSpace((n,)), CoordinateSpace((n,)))

apply_(::ApplyOnlyOp, x) = 2 .* x

struct AdjointOnlyOp <: LinOp{CoordinateSpace{1}, CoordinateSpace{1}}
    inputspace::CoordinateSpace{1}
    outputspace::CoordinateSpace{1}
end

AdjointOnlyOp(n::Int) = AdjointOnlyOp(CoordinateSpace((n,)), CoordinateSpace((n,)))

apply_(::AdjointOnlyOp, x) = 3 .* x
apply_adjoint_(::AdjointOnlyOp, x) = 5 .* x

struct NoApplyOp <: LinOp{CoordinateSpace{1}, CoordinateSpace{1}}
    inputspace::CoordinateSpace{1}
    outputspace::CoordinateSpace{1}
end

NoApplyOp(n::Int) = NoApplyOp(CoordinateSpace((n,)), CoordinateSpace((n,)))

@testset "LinOp - generic mul! fallback via apply_" begin
    A = ApplyOnlyOp(3)
    x = [1.0, 2.0, 3.0]
    y = zeros(3)

    mul!(y, A, x)
    @test y == [2.0, 4.0, 6.0]
end

@testset "LinOp - generic apply errors" begin
    A = NoApplyOp(3)
    x = ones(3)
    y = zeros(3)

    @test_throws ArgumentError A * x
    @test_throws ArgumentError mul!(y, A, x)
end

@testset "LinOpAdjoint - generic fallbacks" begin
    A = AdjointOnlyOp(3)
    x = [1.0, 2.0, 3.0]
    y = zeros(3)

    @test A' * x == [5.0, 10.0, 15.0]

    mul!(y, A', x)
    @test y == [5.0, 10.0, 15.0]
end

@testset "LinOpAdjoint - generic adjoint errors" begin
    A = NoApplyOp(3)
    x = ones(3)

    @test_throws ArgumentError A' * x
end

@testset "Optional operator capability API" begin
    @test LinOps.has_operator(:dft) isa Bool
    @test LinOps.has_operator(:nfft) isa Bool
    @test LinOps.has_operator(:something_else) == false
    @test LinOps.has_operator(LinOps.LinOpDFT) == LinOps.has_operator(:dft)
    @test LinOps.has_operator(LinOps.LinOpNFFT) == LinOps.has_operator(:nfft)

    @test LinOps.operator_backend(:dft) isa Symbol
    @test LinOps.operator_backend(:nfft) isa Symbol
    @test LinOps.operator_backend(:something_else) == :none
    @test LinOps.operator_backend(LinOps.LinOpDFT) == LinOps.operator_backend(:dft)
    @test LinOps.operator_backend(LinOps.LinOpNFFT) == LinOps.operator_backend(:nfft)
end

@testset "Operations simplification branches" begin
    A = ApplyOnlyOp(3)
    D = LinOpDiag([2.0, 3.0, 4.0])

    @test (A / A) === I
    @test (A \ A) === I

    @test (0 * A) == 0
    @test (1 * A) === A

    C = UniformScaling(2) * A
    @test C isa LinOp
    @test (C * [1.0, 2.0, 3.0]) == 4 .* [1.0, 2.0, 3.0]

    @test (1 * D) === D
    @test (0 * D) == UniformScaling(0)
    @test @inferred(D * [1.0, 2.0, 3.0]) == [2.0, 6.0, 12.0]
end

@testset "LinOp display and adjoint forwarding" begin
    D = LinOpDiag([1.0, 2.0, 3.0])
    s = sprint(show, MIME("text/plain"), D)
    @test occursin("Linear Operator:", s)
    @test occursin("LinOpDiag", s)

    AD = D'
    @test occursin("LinOpAdjoint", summary(AD))
    @test inputsize(AD) == outputsize(D)
    @test outputsize(AD) == inputsize(D)
    @test inputspace(AD) == outputspace(D)
    @test outputspace(AD) == inputspace(D)
end
