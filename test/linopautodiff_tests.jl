using Zygote
using FiniteDifferences
using LinOps: LinOp, LinOpDiag, LinOpGrad, LinOpDFT, LinOpMapslice,
    CoordinateSpace, inputsize, outputsize

struct LinOpDiagX{I, D <: AbstractArray} <: LinOp{I, I}
    inputspace::I
    diag::D
end

function LinOpDiagX(diag::D) where {D <: AbstractArray}
    inspace = CoordinateSpace(size(diag))
    return LinOpDiagX(inspace, diag)
end

LinOps.outputspace(A::LinOpDiagX) = LinOps.inputspace(A)
LinOps.isendomorphism(::LinOpDiagX) = true
LinOps.apply_(A::LinOpDiagX, x) = A.diag .* x

# Helper: compare Zygote gradient against a central finite-difference estimate.
function check_gradient(f, x; rtol = 1.0e-4)
    gz = Zygote.gradient(f, x)[1]
    gf = FiniteDifferences.grad(central_fdm(5, 1), f, x)[1]
    return isapprox(gz, gf; rtol = rtol)
end

@testset "LinOpDiag and LinOpMapslice - CPU autodiff (basic tests)" begin
    @testset "LinOpDiag - gradient w.r.t. input" begin
        d = [2.0, 3.0, 4.0]
        A = LinOpDiag(d)
        x = randn(3)
        f(v) = sum(abs2, A * v)
        # Analytical gradient: 2 * (d .* x) .* d
        gz = Zygote.gradient(f, x)[1]
        @test gz ≈ 2 .* (d .* x) .* d
    end

    @testset "LinOpDiag adjoint - gradient w.r.t. input" begin
        d = [2.0, 3.0, 4.0]
        A = LinOpDiag(d)
        x = randn(3)
        f(v) = sum(abs2, A' * v)
        gz = Zygote.gradient(f, x)[1]
        @test gz ≈ 2 .* (d .* x) .* d  # adjoint of real diagonal == diagonal
    end

    @testset "LinOpDiag - gradient matches finite differences" begin
        d = randn(4)
        A = LinOpDiag(d)
        x = randn(4)
        f(v) = sum(abs2, A * v)
        @test check_gradient(f, x)
    end
end

@testset "apply_adjoint_via_ad fallback via Zygote extension" begin
    d = [2.0 3.0; 4.0 5.0]
    D = LinOpDiag(d)
    DX = LinOpDiagX(d)
    x = randn(LinOps.inputspace(DX))

    @test DX * x ≈ D * x
    @test LinOps.apply_adjoint_via_ad(DX, x) ≈ D' * x
    @test DX' * x ≈ D' * x

    f(v) = sum(abs2, D * v)
    g(v) = sum(abs2, DX * v)

    @test Zygote.gradient(f, x)[1] ≈ Zygote.gradient(g, x)[1]
end

@testset "LinOpGrad - autodiff via Zygote" begin
    @testset "LinOpGrad 1D - gradient w.r.t. input" begin
        A = LinOpGrad(5)
        x = randn(5)
        f(v) = sum(abs2, A * v)
        @test check_gradient(f, x)
    end

    @testset "LinOpGrad 2D - gradient w.r.t. input" begin
        A = LinOpGrad((4, 5))
        x = randn(4, 5)
        f(v) = sum(abs2, A * v)
        @test check_gradient(f, x)
    end

    @testset "LinOpGrad 1D adjoint - gradient w.r.t. input" begin
        A = LinOpGrad(5)
        x = randn(outputsize(A)...)
        f(v) = sum(abs2, A' * v)
        @test check_gradient(f, x)
    end
end

@testset "LinOpDFT - autodiff via Zygote" begin
    @testset "LinOpDFT 1D real - gradient w.r.t. input" begin
        A = LinOpDFT(Float64, (8,))
        x = randn(8)
        f(v) = sum(abs2, A * v)
        gz = Zygote.gradient(f, x)[1]
        @test gz isa AbstractArray
        @test all(isfinite, gz)
    end

    @testset "LinOpDFT 1D complex - gradient w.r.t. input" begin
        A = LinOpDFT(ComplexF64, (8,))
        x = randn(ComplexF64, 8)
        f(v) = real(sum(abs2, A * v))
        gz = Zygote.gradient(f, x)[1]
        @test gz isa AbstractArray
        @test !any(isnan, gz)
    end

    @testset "LinOpDFT adjoint - gradient self-consistency" begin
        # rfft/brfft is not a strict Euclidean adjoint pair on half-spectrum,
        # so compare only autodiff well-posedness here.
        A = LinOpDFT(Float64, (8,))
        x = randn(8)
        f(v) = sum(abs2, A' * (A * v))
        gz = Zygote.gradient(f, x)[1]
        @test gz isa AbstractArray
        @test all(isfinite, gz)
    end
end

@testset "LinOpMapslice - autodiff via Zygote" begin
    @testset "LinOpMapslice single LinOp - gradient w.r.t. input" begin
        sz = (3, 5)
        d = randn(5)
        A = LinOpDiag(d)
        M = LinOpMapslice(sz, A; dims = 2)
        x = randn(sz...)
        f(v) = sum(abs2, M * v)
        @test check_gradient(f, x)
    end

    @testset "LinOpMapslice adjoint - gradient w.r.t. input" begin
        sz = (3, 5)
        d = randn(5)
        A = LinOpDiag(d)
        M = LinOpMapslice(sz, A; dims = 2)
        x = randn(sz...)
        f(v) = sum(abs2, M' * v)
        @test check_gradient(f, x)
    end

    @testset "LinOpMapslice 3D - gradient w.r.t. input" begin
        sz = (2, 5, 3)
        d = randn(5)
        A = LinOpDiag(d)
        M = LinOpMapslice(sz, A; dims = 2)
        x = randn(sz...)
        f(v) = sum(abs2, M * v)
        @test check_gradient(f, x)
    end
end

@testset "LinOp compositions - autodiff via Zygote" begin

    @testset "Scalar * LinOpDiag - gradient w.r.t. input" begin
        d = randn(4)
        A = 3.0 * LinOpDiag(d)
        x = randn(4)
        f(v) = sum(abs2, A * v)
        @test check_gradient(f, x)
    end

    @testset "LinOpGrad composed with LinOpDiag - gradient w.r.t. input" begin
        sz = (6,)
        d = randn(outputsize(LinOpGrad(6))...)
        A = LinOpGrad(6)
        W = LinOpDiag(d)
        x = randn(6)
        # f(v) = ||W * (A * v)||^2
        f(v) = sum(abs2, W * (A * v))
        @test check_gradient(f, x)
    end
end
