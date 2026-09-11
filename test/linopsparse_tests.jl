using LinearAlgebra: mul!
using SparseArrays: sparse
using LinOps: LinOp, LinOpSparse, CoordinateSpace, inputspace, outputspace, inputsize, outputsize

@testset "LinOpSparse - properties and forward application" begin
    matrix = sparse(
        [1.0 0.0 2.0 0.0; 0.0 -1.0 0.0 3.0; 4.0 0.0 0.0 0.5],
    )
    A = LinOpSparse(matrix, (2, 2), (3,))
    x = [1.0 2.0; 3.0 4.0]
    expected = reshape(matrix * vec(x), (3,))

    @test A isa LinOp
    @test eltype(A) == Float64
    @test inputspace(A) == CoordinateSpace((2, 2))
    @test outputspace(A) == CoordinateSpace((3,))
    @test inputsize(A) == (2, 2)
    @test outputsize(A) == (3,)
    @test size(A) == ((3,), (2, 2))
    @test A * x == expected
    @test A(x) == expected

    y = similar(expected)
    @test mul!(y, A, x) === y
    @test y == expected
end

@testset "LinOpSparse - adjoint application" begin
    matrix = sparse(
        ComplexF64[1 + im 0.0 2.0; 0.0 -1 + 2im 0.5; 3.0 0.0 -im; 0.0 4.0 1.0],
    )
    A = LinOpSparse(matrix, (3,), (2, 2))
    x = ComplexF64[1 - im 2 + im; 3.0 4 - 2im]
    expected = reshape(matrix' * vec(x), (3,))

    @test A' * x == expected

    y = similar(expected)
    @test mul!(y, A', x) === y
    @test y == expected
    @test A'' == A
end

@testset "LinOpSparse - dimension and domain validation" begin
    matrix = sparse([1.0 0.0; 0.0 1.0])

    @test_throws DimensionMismatch LinOpSparse(matrix, (3,), (2,))
    @test_throws DimensionMismatch LinOpSparse(matrix, (2,), (3,))

    A = LinOpSparse(matrix, (2,), (2,))
    @test_throws ArgumentError A * ones(3)
    @test_throws ArgumentError mul!(zeros(3), A, ones(2))
    @test_throws ArgumentError mul!(zeros(2), A, ones(3))
end
