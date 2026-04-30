using LinearAlgebra: I, UniformScaling, mul!
using LinOps: LinOp, LinOpDiag, CoordinateSpace, TypedCoordinateSpace, inputspace, outputspace, outputtype
import LinOps: apply_, apply_!, apply_adjoint_, apply_adjoint_!

struct ScaleOp{I, O} <: LinOp{I, O}
    inputspace::I
    outputspace::O
    a::Float64
end

_compose_uniform_count(::LinOp) = 0
function _compose_uniform_count(A::LinOps.LinOpCompose)
    left_count = A.left isa UniformScaling ? 1 : _compose_uniform_count(A.left)
    return left_count + _compose_uniform_count(A.right)
end

_compose_uniform_leftmost(::LinOp) = true
function _compose_uniform_leftmost(A::LinOps.LinOpCompose)
    if A.left isa UniformScaling
        return _compose_uniform_count(A.right) == 0
    end
    return _compose_uniform_leftmost(A.left) && _compose_uniform_leftmost(A.right)
end

Base.inv(A::ScaleOp) = ScaleOp(outputspace(A), inputspace(A), 1 / A.a)

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
    @test _compose_uniform_count(Cscaled) <= 1
    @test _compose_uniform_leftmost(Cscaled)

    Czero = UniformScaling(0) * (UniformScaling(3) * D)
    @test Czero == UniformScaling(0)

    Cone = UniformScaling(1) * (UniformScaling(3) * D)
    @test Cone isa LinOp
    @test Cone * x == [6.0, 18.0, 36.0]
    @test _compose_uniform_count(Cone) <= 1
    @test _compose_uniform_leftmost(Cone)

    Cpull = D * (UniformScaling(2) * D)
    @test Cpull isa LinOp
    @test Cpull * x == [8.0, 36.0, 96.0]
    @test _compose_uniform_count(Cpull) <= 1
    @test _compose_uniform_leftmost(Cpull)

    y = similar(x)
    @test @inferred(mul!(y, Cscaled, x)) == [12.0, 36.0, 72.0]
    @test y == [12.0, 36.0, 72.0]

    yadj = similar(x)
    @test @inferred(mul!(yadj, Cscaled', x)) == [12.0, 36.0, 72.0]
    @test yadj == [12.0, 36.0, 72.0]

    @test @inferred(D * (D * x)) == [4.0, 18.0, 48.0]
end

@testset "Operations - LinOpCompose generic branch coverage" begin
    A = ScaleOp(CoordinateSpace((3,)), CoordinateSpace((3,)), 2.0)
    B = ScaleOp(CoordinateSpace((3,)), CoordinateSpace((3,)), 3.0)
    C = ScaleOp(CoordinateSpace((3,)), CoordinateSpace((3,)), 5.0)
    x = [1.0, 2.0, 3.0]

    AB = A * B
    @test @inferred(outputtype(AB, x)) == Float64
    @test @inferred(apply_(AB, x)) == [6.0, 12.0, 18.0]

    y = similar(x)
    @test @inferred(apply_!(y, AB, x)) == [6.0, 12.0, 18.0]
    @test y == [6.0, 12.0, 18.0]

    yadj = similar(x)
    @test @inferred(apply_adjoint_!(yadj, AB, x)) == [6.0, 12.0, 18.0]
    @test yadj == [6.0, 12.0, 18.0]
    @test @inferred(apply_adjoint_(AB, x)) == [6.0, 12.0, 18.0]

    # Exercise compose-over-compose simplification branches.
    AB_us = UniformScaling(2.0) * B
    ABC1 = AB * C
    ABC2 = AB * AB_us
    ABC3 = A * AB_us
    @test _compose_uniform_count(AB_us) <= 1
    @test _compose_uniform_leftmost(AB_us)
    @test _compose_uniform_count(ABC1) <= 1
    @test _compose_uniform_leftmost(ABC1)
    @test _compose_uniform_count(ABC2) <= 1
    @test _compose_uniform_leftmost(ABC2)
    @test _compose_uniform_count(ABC3) <= 1
    @test _compose_uniform_leftmost(ABC3)
    @test ABC1 * x == [30.0, 60.0, 90.0]
    @test ABC2 * x == [36.0, 72.0, 108.0]
    @test ABC3 * x == [12.0, 24.0, 36.0]

    # UniformScaling-on-compose branch with 0/1/non-trivial factors.
    @test UniformScaling(0.0) * AB_us == 0
    @test @inferred((UniformScaling(0.5) * AB_us) * x) == [3.0, 6.0, 9.0]
    @test @inferred((UniformScaling(1.0) * AB_us) * x) == [6.0, 12.0, 18.0]

    # Scalar compose constructor branch.
    @test @inferred((2 * B) * x) == [6.0, 12.0, 18.0]

    # Direct UniformScaling-over-LinOp constructor branch.
    @test (UniformScaling(0.0) * B) == 0
    @test (UniformScaling(1.0) * B) === B

    # Specialized in-place apply for UniformScaling compose.
    yus = similar(x)
    @test @inferred(apply_!(yus, UniformScaling(2.0) * B, x)) == [6.0, 12.0, 18.0]
    @test yus == [6.0, 12.0, 18.0]

    # Inverse and power branches.
    @test @inferred(inv(AB) * x) == (1 / 6.0) .* x
    @test (B^0) == I
    @test @inferred((B^2) * x) == [9.0, 18.0, 27.0]
end

@testset "Operations - LinOpSum and mixed divide/solve overloads" begin
    A = LinOpDiag([2.0, 3.0, 4.0])
    B = LinOpDiag([5.0, 6.0, 7.0])
    x = [1.0, 2.0, 3.0]

    S = A + B
    @test @inferred(S * x) == (A * x) + (B * x)

    y = similar(x)
    @test @inferred(mul!(y, S, x)) == (A * x) + (B * x)
    @test y == (A * x) + (B * x)

    yadj = similar(x)
    @test @inferred(mul!(yadj, S', x)) == (A' * x) + (B' * x)
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

@testset "Operations - LinOpSum generic apply branches" begin
    A = ScaleOp(CoordinateSpace((3,)), CoordinateSpace((3,)), 2.0)
    B = ScaleOp(CoordinateSpace((3,)), CoordinateSpace((3,)), 3.0)
    x = [1.0, 2.0, 3.0]

    S = A + B
    @test @inferred(apply_(S, x)) == [5.0, 10.0, 15.0]

    y = similar(x)
    @test @inferred(apply_!(y, S, x)) == [5.0, 10.0, 15.0]
    @test y == [5.0, 10.0, 15.0]

    @test @inferred(apply_adjoint_(S, x)) == [5.0, 10.0, 15.0]
    yadj = similar(x)
    @test @inferred(apply_adjoint_!(yadj, S, x)) == [5.0, 10.0, 15.0]
    @test yadj == [5.0, 10.0, 15.0]

    # Number/UniformScaling sum constructor branches.
    @test (0 + B) === B
    @test (UniformScaling(0) + B) === B
    @test @inferred((2 + B) * x) == [5.0, 10.0, 15.0]
    @test @inferred((UniformScaling(2) + B) * x) == [5.0, 10.0, 15.0]
end

@testset "Operations - LinOpCompose UniformScaling invariant" begin
    A = ScaleOp(CoordinateSpace((3,)), CoordinateSpace((3,)), 2.0)
    B = ScaleOp(CoordinateSpace((3,)), CoordinateSpace((3,)), 3.0)
    C = ScaleOp(CoordinateSpace((3,)), CoordinateSpace((3,)), 5.0)

    K1 = UniformScaling(2.0) * (UniformScaling(3.0) * A)
    K2 = (A * (UniformScaling(2.0) * B)) * (UniformScaling(4.0) * C)
    K3 = UniformScaling(1.0) * (A * (UniformScaling(2.0) * B))

    @test _compose_uniform_count(K1) <= 1
    @test _compose_uniform_leftmost(K1)
    @test _compose_uniform_count(K2) <= 1
    @test _compose_uniform_leftmost(K2)
    @test _compose_uniform_count(K3) <= 1
    @test _compose_uniform_leftmost(K3)
end
