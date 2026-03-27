@testset "Domains - CoordinateSpace Construction" begin
    # Single dimension
    sp1 = CoordinateSpace(10)
    @test size(sp1) == (10,)
    @test ndims(sp1) == 1

    # Multiple dimensions
    sp2 = CoordinateSpace((5, 8, 3))
    @test size(sp2) == (5, 8, 3)
    @test ndims(sp2) == 3

    # Zero-dimensional
    sp0 = CoordinateSpace()
    @test size(sp0) == ()
    @test ndims(sp0) == 0
    @test length(sp0) == 1

    # Copy construction
    sp_copy = CoordinateSpace(sp2)
    @test sp_copy === sp2
    @test size(sp_copy) == size(sp2)
end

@testset "Domains - Size and Dimensionality" begin
    sp = CoordinateSpace((4, 6, 2))

    # Size queries
    @test size(sp) == (4, 6, 2)
    @test size(sp, 1) == 4
    @test size(sp, 2) == 6
    @test size(sp, 3) == 2
    @test size(sp, 4) == 1  # Beyond dimensions returns 1
    @test size(sp, 100) == 1

    # Dimensionality
    @test ndims(sp) == 3
    @test ndims(CoordinateSpace((10, 20))) == 2
    @test ndims(CoordinateSpace(5)) == 1

    # Type-based ndims
    @test ndims(CoordinateSpace{3}) == 3
    @test ndims(CoordinateSpace{1}) == 1
end

@testset "Domains - Axes" begin
    sp = CoordinateSpace((3, 4, 5))

    # Axes for each dimension
    @test axes(sp, 1) == Base.OneTo(3)
    @test axes(sp, 2) == Base.OneTo(4)
    @test axes(sp, 3) == Base.OneTo(5)

    # Beyond dimensions returns OneTo(1)
    @test axes(sp, 4) == Base.OneTo(1)
end

@testset "Domains - Length and Total Size" begin
    sp1 = CoordinateSpace((3, 4))
    @test length(sp1) == 12

    sp2 = CoordinateSpace((2, 3, 5))
    @test length(sp2) == 30

    sp3 = CoordinateSpace(7)
    @test length(sp3) == 7

    sp0 = CoordinateSpace()
    @test length(sp0) == 1
end

@testset "Domains - Membership Testing" begin
    sp = CoordinateSpace((3, 4))

    # Arrays with matching size
    @test zeros(3, 4) in sp
    @test ones(3, 4) in sp
    @test rand(3, 4) in sp

    # Wrong size
    @test !(zeros(3, 5) in sp)
    @test !(zeros(4, 3) in sp)
    @test !(zeros(3) in sp)

    # Different types but correct size
    @test zeros(Int, 3, 4) in sp
    @test zeros(Float64, 3, 4) in sp
    @test zeros(ComplexF64, 3, 4) in sp

    # Different dimensional arrays
    sp1d = CoordinateSpace(10)
    @test zeros(10) in sp1d
    @test !(zeros(10, 1) in sp1d)
    @test !(zeros(5, 2) in sp1d)
end

@testset "Domains - Array Creation Methods" begin
    sp = CoordinateSpace((2, 3))

    # zeros
    z = zeros(sp)
    @test size(z) == (2, 3)
    @test all(iszero, z)
    @test eltype(z) == Float64

    z_int = zeros(Int, sp)
    @test size(z_int) == (2, 3)
    @test all(iszero, z_int)
    @test eltype(z_int) == Int

    # ones
    o = ones(sp)
    @test size(o) == (2, 3)
    @test all(isone, o)
    @test eltype(o) == Float64

    o_int = ones(Int32, sp)
    @test size(o_int) == (2, 3)
    @test all(isone, o_int)
    @test eltype(o_int) == Int32

    # randn (more reliable than rand for type testing)
    rn = randn(sp)
    @test size(rn) == (2, 3)
    @test eltype(rn) == Float64

    rn_float32 = randn(Float32, sp)
    @test size(rn_float32) == (2, 3)
    @test eltype(rn_float32) == Float32

    # rand (returns tuple when called with tuple argument)
    # So we test with unpacked size instead
    r = rand(sp)
    @test size(r) == (2, 3)
    @test eltype(r) == Float64
end

@testset "Domains - Similar" begin
    sp = CoordinateSpace((3, 4))

    # Create similar from different source arrays
    A = zeros(5, 6)
    B = similar(A, sp)
    @test size(B) == (3, 4)
    @test eltype(B) == eltype(A)  # Preserves element type

    # Different element types
    C = ones(Float32, 2, 2)
    D = similar(C, sp)
    @test size(D) == (3, 4)
    @test eltype(D) == Float32

    # Test with various source arrays
    E = collect(1:12)
    F = similar(E, sp)
    @test size(F) == (3, 4)
    @test eltype(F) == typeof(1)
end

@testset "Domains - Edge Cases" begin
    # Single element domain
    sp_single = CoordinateSpace(1)
    @test size(sp_single) == (1,)
    @test length(sp_single) == 1
    @test zeros(sp_single) in sp_single

    # Large dimensions
    sp_large = CoordinateSpace((100, 200))
    @test size(sp_large) == (100, 200)
    @test length(sp_large) == 20000

    # Constructor with single integer
    sp = CoordinateSpace(42)
    @test size(sp) == (42,)
    @test ndims(sp) == 1
    @test length(sp) == 42
end
