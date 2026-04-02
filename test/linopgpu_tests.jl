using GPUArrays
using JLArrays
using LinearAlgebra: mul!
using LinOps: LinOp, LinOpDiag, LinOpMapslice, inputsize, outputsize, CoordinateSpace, inputspace, outputspace

@testset "LinOpDiag - GPU arrays" begin
    @testset "LinOpDiag forward pass on GPU" begin
        # CPU reference
        d_cpu = [2.0, 3.0, 4.0]
        x_cpu = randn(3)
        A_cpu = LinOpDiag(d_cpu)
        y_cpu = A_cpu * x_cpu

        # GPU variant
        d_gpu = JLArray(d_cpu)
        x_gpu = JLArray(x_cpu)
        A_gpu = LinOpDiag(d_gpu)
        y_gpu = A_gpu * x_gpu

        @test y_gpu ≈ JLArray(y_cpu)
        @test Array(y_gpu) ≈ y_cpu
    end

    @testset "LinOpDiag forward pass on GPU - 2D diagonal" begin
        # CPU reference
        d_cpu = Float32[2.0 3.0; 4.0 5.0]
        x_cpu = randn(Float32, 2, 2)
        A_cpu = LinOpDiag(d_cpu)
        y_cpu = A_cpu * x_cpu

        # GPU variant
        d_gpu = JLArray(d_cpu)
        x_gpu = JLArray(x_cpu)
        A_gpu = LinOpDiag(d_gpu)
        y_gpu = A_gpu * x_gpu

        @test y_gpu ≈ JLArray(y_cpu)
        @test Array(y_gpu) ≈ y_cpu rtol = 1.0f-5
    end

    @testset "LinOpDiag mul! on GPU" begin
        # CPU reference
        d_cpu = [2.0, 3.0, 4.0]
        x_cpu = randn(3)
        y_cpu = zeros(3)
        A_cpu = LinOpDiag(d_cpu)
        mul!(y_cpu, A_cpu, x_cpu)

        # GPU variant
        d_gpu = JLArray(d_cpu)
        x_gpu = JLArray(x_cpu)
        y_gpu = JLArray(zeros(3))
        A_gpu = LinOpDiag(d_gpu)
        mul!(y_gpu, A_gpu, x_gpu)

        @test Array(y_gpu) ≈ y_cpu
    end

    @testset "LinOpDiag adjoint on GPU - complex" begin
        # CPU reference
        d_cpu = ComplexF64[1 + 2im, 2 - im, 3 + 0im]
        x_cpu = randn(ComplexF64, 3)
        A_cpu = LinOpDiag(d_cpu)
        y_cpu = A_cpu' * x_cpu

        # GPU variant
        d_gpu = JLArray(d_cpu)
        x_gpu = JLArray(x_cpu)
        A_gpu = LinOpDiag(d_gpu)
        y_gpu = A_gpu' * x_gpu

        @test y_gpu ≈ JLArray(y_cpu)
        @test Array(y_gpu) ≈ y_cpu rtol = 1.0e-12
    end

    @testset "LinOpDiag adjoint mul! on GPU - complex" begin
        # CPU reference
        d_cpu = ComplexF64[1 + 2im, 2 - im, 3 + 0im]
        x_cpu = randn(ComplexF64, 3)
        y_cpu = similar(x_cpu)
        A_cpu = LinOpDiag(d_cpu)
        mul!(y_cpu, A_cpu', x_cpu)

        # GPU variant
        d_gpu = JLArray(d_cpu)
        x_gpu = JLArray(x_cpu)
        y_gpu = JLArray(similar(x_cpu))
        A_gpu = LinOpDiag(d_gpu)
        mul!(y_gpu, A_gpu', x_gpu)

        @test Array(y_gpu) ≈ y_cpu rtol = 1.0e-12
    end

    @testset "LinOpDiag composition on GPU" begin
        # CPU reference
        d1_cpu = [2.0, 3.0]
        d2_cpu = [4.0, 5.0]
        x_cpu = randn(2)
        A_cpu = LinOpDiag(d1_cpu)
        B_cpu = LinOpDiag(d2_cpu)
        C_cpu = A_cpu * B_cpu
        y_cpu = C_cpu * x_cpu

        # GPU variant
        d1_gpu = JLArray(d1_cpu)
        d2_gpu = JLArray(d2_cpu)
        x_gpu = JLArray(x_cpu)
        A_gpu = LinOpDiag(d1_gpu)
        B_gpu = LinOpDiag(d2_gpu)
        C_gpu = A_gpu * B_gpu
        y_gpu = C_gpu * x_gpu

        @test y_gpu ≈ JLArray(y_cpu)
        @test Array(y_gpu) ≈ y_cpu
    end

    @testset "LinOpDiag properties on GPU" begin
        d_gpu = JLArray(Float32[2.0, 3.0, 4.0])
        A_gpu = LinOpDiag(d_gpu)

        @test inputspace(A_gpu) == CoordinateSpace((3,))
        @test outputspace(A_gpu) == inputspace(A_gpu)
        @test inputsize(A_gpu) == (3,)
        @test outputsize(A_gpu) == (3,)
        @test eltype(A_gpu) == Float32
    end

    @testset "LinOpDiag on GPU with 3D arrays" begin
        # CPU reference
        d_cpu = randn(3, 4, 5)
        x_cpu = randn(3, 4, 5)
        A_cpu = LinOpDiag(d_cpu)
        y_cpu = A_cpu * x_cpu

        # GPU variant
        d_gpu = JLArray(d_cpu)
        x_gpu = JLArray(x_cpu)
        A_gpu = LinOpDiag(d_gpu)
        y_gpu = A_gpu * x_gpu

        @test Array(y_gpu) ≈ y_cpu rtol = 1.0e-12
    end
end

@testset "LinOpMapslice - GPU arrays (simple single LinOp case)" begin
    @testset "LinOpMapslice single LinOp on CPU arrays first (reference)" begin
        # Ensure CPU version works as reference
        sz = (2, 5, 3)
        d_cpu = randn(5)
        x_cpu = randn(sz...)
        D_cpu = LinOpDiag(d_cpu)
        M_cpu = LinOpMapslice(sz, D_cpu, 2)
        y_cpu = similar(x_cpu)

        # Direct computation without KernelAbstractions
        for i in axes(x_cpu, 1), k in axes(x_cpu, 3)
            y_cpu[i, :, k] .= D_cpu * view(x_cpu, i, :, k)
        end

        # Using operator
        y_op = M_cpu * x_cpu
        @test y_op ≈ y_cpu rtol = 1.0e-12
    end

    @testset "LinOpMapslice adjoint on CPU - complex (reference)" begin
        sz = (2, 5, 3)
        d_cpu = randn(ComplexF64, 5)
        x_cpu = randn(ComplexF64, sz...)
        D_cpu = LinOpDiag(d_cpu)
        M_cpu = LinOpMapslice(sz, D_cpu, 2)

        y_expected = similar(x_cpu)
        for i in axes(x_cpu, 1), k in axes(x_cpu, 3)
            y_expected[i, :, k] .= D_cpu' * view(x_cpu, i, :, k)
        end

        y_op = M_cpu' * x_cpu
        @test y_op ≈ y_expected rtol = 1.0e-12
    end

    @testset "LinOpMapslice properties on GPU" begin
        sz = (2, 5, 3)
        d_gpu = JLArray(randn(5))
        D_gpu = LinOpDiag(d_gpu)
        M_gpu = LinOpMapslice(sz, D_gpu, 2)

        @test inputsize(M_gpu) == sz
        @test outputsize(M_gpu) == sz
        @test inputspace(M_gpu) == CoordinateSpace(sz)
        @test outputspace(M_gpu) == CoordinateSpace(sz)
    end
end
