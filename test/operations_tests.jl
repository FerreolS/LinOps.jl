using LinearAlgebra: I, UniformScaling, mul!
using LinOps: LinOp, LinOpDiag, CoordinateSpace, TypedCoordinateSpace, inputspace, outputspace
import LinOps: apply_, apply_!, apply_adjoint_, apply_adjoint_!

struct ScaleOp{I, O} <: LinOp{I, O}
    inputspace::I
    outputspace::O
    a::Float64
end

apply_(A::ScaleOp, x) = A.a .* x
apply_!(y, A::ScaleOp, x) = (@. y = A.a * x)
apply_adjoint_(A::ScaleOp, x) = A.a .* x
apply_adjoint_!(y, A::ScaleOp, x) = (@. y = A.a * x)

@testset "Operations - LinOpCompose domain promotion branches" begin
    left_typed = ScaleOp(TypedCoordinateSpace(Float32, (3,)), CoordinateSpace((3,)), 2.0)
    right_coord = ScaleOp(CoordinateSpace((3,)), CoordinateSpace((3,)), 3.0)
    C1 = left_typed * right_coord
    @test inputspace(C1) == TypedCoordinateSpace(Float32, (3,))

    left_coord = ScaleOp(CoordinateSpace((3,)), CoordinateSpace((3,)), 2.0)
    right_typed_out = ScaleOp(CoordinateSpace((3,)), TypedCoordinateSpace(Float32, (3,)), 3.0)
    C2 = left_coord * right_typed_out
    @test outputspace(C2) == TypedCoordinateSpace(Float32, (3,))
end

@testset "Operations - LinOpCompose simplify and in-place paths" begin
    D = LinOpDiag([2.0, 3.0, 4.0])
    x = [1.0, 2.0, 3.0]

    Cscaled = UniformScaling(2) * (UniformScaling(3) * D)
    @test Cscaled isa LinOp
    @test Cscaled * x == [12.0, 36.0, 72.0]

    Czero = UniformScaling(0) * (UniformScaling(3) * D)
    @test Czero == UniformScaling(0)

    Cone = UniformScaling(1) * (UniformScaling(3) * D)
    @test Cone isa LinOp
    @test Cone * x == [6.0, 18.0, 36.0]

    Cpull = D * (UniformScaling(2) * D)
    @test Cpull isa LinOp
    @test Cpull * x == [8.0, 36.0, 96.0]

    y = similar(x)
    mul!(y, Cscaled, x)
    @test y == [12.0, 36.0, 72.0]

    yadj = similar(x)
    mul!(yadj, Cscaled', x)
    @test yadj == [12.0, 36.0, 72.0]

    @test @inferred(D * (D * x)) == [4.0, 18.0, 48.0]
end

@testset "Operations - LinOpSum and mixed divide/solve overloads" begin
    A = LinOpDiag([2.0, 3.0, 4.0])
    B = LinOpDiag([5.0, 6.0, 7.0])
    x = [1.0, 2.0, 3.0]

    S = A + B
    y = similar(x)
    mul!(y, S, x)
    @test y == (A * x) + (B * x)

    yadj = similar(x)
    mul!(yadj, S', x)
    @test yadj == (A' * x) + (B' * x)

    @test (2 / A) isa LinOp
    @test (A / 2) isa LinOp
    @test (2 \ A) isa LinOp
    @test (A \ 2) isa LinOp
    @test (A / A) == I
    @test (A \ A) == I

    @test (-A) isa LinOp
    @test (A - B) isa LinOp
    @test (2 - A) isa LinOp
    @test (A - 2) isa LinOp

    @test @inferred(A * x) == [2.0, 6.0, 12.0]
    @test @inferred(S * x) == (A * x) + (B * x)
end

@testset "Operations - LinOpSum constructor message checks" begin
    A = LinOpDiag([2.0, 3.0, 4.0])
    B_bad_in = ScaleOp(CoordinateSpace((2,)), CoordinateSpace((3,)), 1.0)
    B_bad_out = ScaleOp(CoordinateSpace((3,)), CoordinateSpace((2,)), 1.0)

    @test_throws "input spaces of the two operators should match" A + B_bad_in
    @test_throws "output spaces of the two operators should match" A + B_bad_out
end
