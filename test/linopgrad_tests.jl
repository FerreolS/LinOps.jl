using LinearAlgebra: mul!
using LinOps: LinOp, LinOpGrad, CoordinateSpace, inputsize, outputsize, inputspace, outputspace

@testset "LinOpGrad - Basic LinOp properties" begin
    A = LinOpGrad((5, 4))

    @test A isa LinOp
    @test inputspace(A) == CoordinateSpace((5, 4))
    @test outputspace(A) == CoordinateSpace((5, 4, 2))
    @test inputsize(A) == (5, 4)
    @test outputsize(A) == (5, 4, 2)
    @test size(A) == ((5, 4, 2), (5, 4))
end

@testset "LinOpGrad - 1D forward and mul!" begin
    x = Float64[1, 4, 9, 16, 25]
    A = LinOpGrad(length(x))

    y = A * x
    expected = zeros(Float64, length(x), 1)
    for i in 1:(length(x) - 1)
        expected[i, 1] = x[i] - x[i + 1]
    end

    @test y == expected

    y2 = similar(y)
    mul!(y2, A, x)
    @test y2 == expected
end

@testset "LinOpGrad - 1D adjoint and mul!" begin
    x = Float64[1, 4, 9, 16, 25]
    A = LinOpGrad(length(x))
    g = reshape(Float64[2, -1, 3, 5, 99], :, 1)

    y = A' * g
    expected = zeros(Float64, length(x))
    for i in 1:(length(x) - 1)
        expected[i] += g[i, 1]
        expected[i + 1] -= g[i, 1]
    end
    @test y == expected

    y2 = similar(expected)
    mul!(y2, A', g)
    @test y2 == expected
end

@testset "LinOpGrad - 2D forward" begin
    x = reshape(Float64.(1:12), 3, 4)
    A = LinOpGrad(size(x))

    y = A * x
    expected = zeros(Float64, 3, 4, 2)

    for j in 1:size(x, 2)
        for i in 1:(size(x, 1) - 1)
            expected[i, j, 1] = x[i, j] - x[i + 1, j]
        end
    end
    for j in 1:(size(x, 2) - 1)
        for i in 1:size(x, 1)
            expected[i, j, 2] = x[i, j] - x[i, j + 1]
        end
    end

    @test y == expected
end

@testset "LinOpGrad - Adjointness relation" begin
    x = randn(6, 5)
    A = LinOpGrad(size(x))
    y = randn(outputsize(A)...)

    lhs = sum((A * x) .* y)
    rhs = sum(x .* (A' * y))

    @test lhs ≈ rhs rtol = 1.0e-12 atol = 1.0e-12
end

@testset "LinOpGrad - Custom offsets " begin
    x = randn(3, 4, 5)
    A = LinOpGrad(size(x); offsets = [1, 2, 0])
    y = A * x

    @test outputsize(A) == (3, 4, 5, 2)
    @test size(y) == (3, 4, 5, 2)

    expected = zeros(size(y))
    for i in 1:(size(x, 1) - 1), j in 1:size(x, 2), k in 1:size(x, 3)
        expected[i, j, k, 1] = x[i, j, k] - x[i + 1, j, k]
    end
    for i in 1:size(x, 1), j in 1:(size(x, 2) - 2), k in 1:size(x, 3)
        expected[i, j, k, 2] = x[i, j, k] - x[i, j + 2, k]
    end

    @test y ≈ expected
end

@testset "LinOpGrad - Zero offsets dimensions" begin
    x = randn(4, 3)
    A = LinOpGrad(size(x); offsets = [0, 1])
    y = A * x

    @test outputsize(A) == (4, 3, 1)
    @test size(y) == (4, 3, 1)

    expected = zeros(4, 3, 1)
    for i in 1:4, j in 1:2
        expected[i, j, 1] = x[i, j] - x[i, j + 1]
    end
    @test y ≈ expected
end

@testset "LinOpGrad - Custom offsets validation" begin
    x = randn(3, 4, 5)
    @test_throws "LinOpGrad offsets must have length 3" LinOpGrad(size(x); offsets = [1, 2])
    @test_throws "LinOpGrad offsets offsets must be >= 0" LinOpGrad(size(x); offsets = [1, -1, 0])
    @test_throws "LinOpGrad offsets offset along dimension 1 is 4 but size is 3" LinOpGrad(size(x); offsets = [4, 0, 0])

    @test LinOpGrad(size(x); offsets = [1, 1, 0]) isa LinOp
end
