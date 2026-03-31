using LinearAlgebra: mul!
using LinOps: LinOpMapslice, LinOpDiag, inputsize, outputsize

@testset "LinOpMapslice - single LinOp operator" begin
    sz = (2, 5, 3)
    d = randn(5)
    D = LinOpDiag(d)
    M = LinOpMapslice(sz, D, 2)

    @test inputsize(M) == sz
    @test outputsize(M) == sz

    x = randn(sz...)
    y = M * x

    expected = similar(x)
    for i in axes(x, 1), k in axes(x, 3)
        expected[i, :, k] .= D * view(x, i, :, k)
    end

    @test y ≈ expected
    @test M(x) ≈ expected

    y2 = similar(x)
    mul!(y2, M, x)
    @test y2 ≈ expected
end

@testset "LinOpMapslice - single LinOp adjoint" begin
    sz = (2, 5, 3)
    d = randn(ComplexF64, 5)
    D = LinOpDiag(d)
    M = LinOpMapslice(sz, D, 2)

    x = randn(ComplexF64, sz...)
    y = M' * x

    expected = similar(x)
    for i in axes(x, 1), k in axes(x, 3)
        expected[i, :, k] .= D' * view(x, i, :, k)
    end

    @test y ≈ expected

    y2 = similar(x)
    mul!(y2, M', x)
    @test y2 ≈ expected
end

@testset "LinOpMapslice - array of LinOp operators" begin
    sz = (2, 5, 3)
    ops = [LinOpDiag(randn(5)) for _ in 1:sz[1], _ in 1:sz[3]]
    M = LinOpMapslice(sz, ops, 2)

    x = randn(sz...)
    y = M * x

    expected = similar(x)
    for i in axes(x, 1), k in axes(x, 3)
        expected[i, :, k] .= ops[i, k] * view(x, i, :, k)
    end

    @test y ≈ expected

    y_adj = M' * x
    expected_adj = similar(x)
    for i in axes(x, 1), k in axes(x, 3)
        expected_adj[i, :, k] .= ops[i, k]' * view(x, i, :, k)
    end

    @test y_adj ≈ expected_adj
end

@testset "LinOpMapslice - array of Matrices and array of number" begin
    sz = (2, 5, 3)

    mats = [randn(3, 5) for _ in 1:sz[1], _ in 1:sz[3]]
    Mmat = LinOpMapslice(sz, mats, 2)

    x = randn(sz...)
    y = Mmat * x
    @test size(y) == (2, 3, 3)

    expected_mat = similar(y)
    for i in axes(x, 1), k in axes(x, 3)
        expected_mat[i, :, k] .= mats[i, k] * view(x, i, :, k)
    end
    @test y ≈ expected_mat

    scalars = randn(ComplexF64, sz[1], sz[3])
    Mnum = LinOpMapslice(sz, scalars, 2)

    x2 = randn(ComplexF64, sz...)
    y2 = Mnum * x2
    expected_num = similar(x2)
    for i in axes(x2, 1), k in axes(x2, 3)
        expected_num[i, :, k] .= scalars[i, k] .* view(x2, i, :, k)
    end
    @test y2 ≈ expected_num

    y2_adj = Mnum' * x2
    expected_num_adj = similar(x2)
    for i in axes(x2, 1), k in axes(x2, 3)
        expected_num_adj[i, :, k] .= conj(scalars[i, k]) .* view(x2, i, :, k)
    end
    @test y2_adj ≈ expected_num_adj

    us = [complex(randn(), randn()) * I for _ in 1:sz[1], _ in 1:sz[3]]
    Mus = LinOpMapslice(sz, us, 2)

    y3 = Mus * x2
    expected_us = similar(x2)
    for i in axes(x2, 1), k in axes(x2, 3)
        expected_us[i, :, k] .= us[i, k] * view(x2, i, :, k)
    end
    @test y3 ≈ expected_us

    y3_adj = Mus' * x2
    expected_us_adj = similar(x2)
    for i in axes(x2, 1), k in axes(x2, 3)
        expected_us_adj[i, :, k] .= us[i, k]' * view(x2, i, :, k)
    end
    @test y3_adj ≈ expected_us_adj
end

@testset "LinOpMapslice - argument checks" begin
    @test_throws ArgumentError LinOpMapslice((2, 5, 3), LinOpDiag(randn(4)), [2])
    @test_throws ArgumentError LinOpMapslice((2, 5, 3), LinOpDiag(randn(5)), [4])

    ops = [LinOpDiag(randn(5)) for _ in 1:2, _ in 1:2]
    @test_throws ArgumentError LinOpMapslice((2, 5, 3), ops, [2])
end
