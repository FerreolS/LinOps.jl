using LinearAlgebra: mul!
using LinOps: LinOp, CoordinateSpace
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
