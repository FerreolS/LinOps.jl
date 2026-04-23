using LinearAlgebra: I, UniformScaling, mul!
using FixedSizeArrays: FixedSizeArrayDefault
using Adapt: adapt
using LinOps: LinOp, LinOpDiag, CoordinateSpace, inputspace, outputspace, inputsize, outputsize, isendomorphism

@testset "LinOpDiag - Basic LinOp properties" begin
    d = [2.0 3.0; 4.0 5.0]
    x = ones(2, 2)
    A = LinOpDiag(d)
    Aadj = A'

    @test A isa LinOp
    @test isendomorphism(A) == true
    @test inputspace(A) == CoordinateSpace((2, 2))
    @test outputspace(A) == inputspace(A)
    @test inputsize(A) == (2, 2)
    @test outputsize(A) == (2, 2)
    @test size(A) == ((2, 2), (2, 2))
    @test isendomorphism(A)
    @test isendomorphism(I)
    @test LinOps.outputtype(A, x) == Float64
    @test LinOps.outputtype(I, x) == Float64

    @test inputspace(Aadj) == outputspace(A)
    @test outputspace(Aadj) == inputspace(A)
end

@testset "LinOpDiag - Adapt structure" begin
    A = LinOpDiag([1.0, 2.0])
    D = FixedSizeArrayDefault([1.0, 2.0])
    B = adapt(FixedSizeArrayDefault, A)

    @test B isa LinOpDiag
    @test B.diag == D
    @test B.diag isa typeof(D)
    @test inputspace(B) == inputspace(A)
    @test inputspace(B) == CoordinateSpace((2,))
end

@testset "LinOpDiag - Apply via * and call" begin
    d = [2.0 3.0; 4.0 5.0]
    x = [1.0 2.0; 3.0 4.0]
    A = LinOpDiag(d)

    y = A * x
    @test y == d .* x
    @test A(x) == y

    y2 = similar(x)
    mul!(y2, A, x)
    @test y2 == d .* x
end

@testset "LinOpDiag - Domain and codomain checks" begin
    d = [1.0 2.0; 3.0 4.0]
    A = LinOpDiag(d)

    x_bad = zeros(2)
    @test_throws ArgumentError A * x_bad

    x = zeros(2, 2)
    y_bad = zeros(2)
    @test_throws ArgumentError mul!(y_bad, A, x)
end

@testset "LinOpDiag - Adjoint behavior" begin
    d = ComplexF64[1 + 2im 2 - im; 3 + 0im 4 - 3im]
    x = ComplexF64[2 - im 0 + 2im; -1 + 0im 1 - im]
    A = LinOpDiag(d)

    @test A' * x == conj.(d) .* x

    y = similar(x)
    mul!(y, A', x)
    @test y == conj.(d) .* x

    @test A'' == A
end

@testset "LinOpDiag - Operations from Operations.jl" begin
    d1 = [2.0 3.0; 4.0 5.0]
    d2 = [7.0 11.0; 13.0 17.0]
    x = [1.0 2.0; 3.0 4.0]

    A = LinOpDiag(d1)
    B = LinOpDiag(d2)

    C = A * B
    @test C isa LinOpDiag
    @test C * x == (d1 .* d2) .* x

    D = A ∘ B
    @test D isa LinOpDiag
    @test D * x == (d1 .* d2) .* x

    E = I ∘ B
    @test E isa LinOpDiag
    @test E == B

    E = B ∘ I
    @test E isa LinOpDiag
    @test E == B

    S = A + B
    @test S isa LinOpDiag
    @test S * x == (d1 .+ d2) .* x

    S2 = I + B
    @test S2 isa LinOpDiag
    @test S2 * x == (1 .+ d2) .* x

    S3 = 3 + A
    @test S3 isa LinOpDiag
    @test S3 * x == (3 .+ d1) .* x

    C3 = 3 * A
    @test C3 isa LinOpDiag
    @test C3 * x == (3 .* d1) .* x

    C3r = A * 3
    @test C3r isa LinOpDiag
    @test C3r * x == (3 .* d1) .* x

    S3r = A + 3
    @test S3r isa LinOpDiag
    @test S3r * x == (3 .+ d1) .* x

    C0 = 0 * A
    @test C0 == UniformScaling(0)
    @test C0 * x == zero.(x)

    Cu = UniformScaling(3) * A
    @test Cu isa LinOpDiag
    @test Cu * x == (3 .* d1) .* x

    @test (A^2) isa LinOpDiag
    @test (A^2) * x == (d1 .^ 2) .* x

    Ainv = inv(A)
    @test Ainv isa LinOpDiag
    @test Ainv * x ≈ (1.0 ./ d1) .* x

    @test (A / A) === I
    @test (A \ A) === I

    D = A / B
    @test D isa LinOpDiag
    @test D * x ≈ ((d1 ./ d2) .* x)

    L = A \ B
    @test L isa LinOpDiag
    @test L * x ≈ ((d2 ./ d1) .* x)
end

@testset "LinOpDiag - Power identity behavior" begin
    d = [2.0 4.0; 8.0 16.0]
    x = [1.0 2.0; 3.0 4.0]
    A = LinOpDiag(d)

    @test (A^0) isa LinOpDiag
    @test (A^0) * x == x
end

@testset "LinOpDiag - Identity and cancellation branches" begin
    d = [2.0 4.0; 8.0 16.0]
    x = [1.0 2.0; 3.0 4.0]
    A = LinOpDiag(d)
    Ainv = inv(A)

    @test (A * Ainv) === I
    @test (Ainv * A) === I

    @test (1 * A) === A

    Z = UniformScaling(0) * A
    @test Z == UniformScaling(0)
    @test Z * x == zero.(x)

    S0 = A + LinOpDiag(-d)
    @test S0 == UniformScaling(0)
    @test S0 * x == zero.(x)
end
