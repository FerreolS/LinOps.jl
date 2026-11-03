using InterpolationKernels
using LinearAlgebra: dot
using LinOps
using LinOps: LinOpInterpolator, verify_adjoint

@testset "LinOpInterpolator" begin
    knots = 0.0:0.5:4.0
    x = sin.(knots)
    p = [0.1 1.3; 2.2 5.0; -1.0 3.5]
    Ls = LinOpInterpolator(knots, p)
    @test Ls isa LinOps.LinOpSparse
    Lf = LinOpInterpolator(knots, p; sparse = false)
    @test Ls * x ≈ Lf * x
    @test size(Ls * x) == size(p)
    @test (Ls * x)[1, 2] ≈ sin(1.3) atol = 1.0e-2
    @test (Ls * x)[2, 2] ≈ sin(4.0)
    y = randn(size(p))
    @test Ls' * y ≈ Lf' * y
    @test verify_adjoint(Ls)
    @test verify_adjoint(Lf)

    K = LinOpInterpolator(knots, p, BSpline{4}())
    @test K * ones(length(knots)) ≈ ones(size(p))
end
