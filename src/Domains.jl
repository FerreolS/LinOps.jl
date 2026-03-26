abstract type AbstractDomain end

struct CoordinateSpace{N} <: AbstractDomain
    size::NTuple{N, Int}
end
#=
struct TypedCoordinateSpace{N} <: AbstractDomain
    type::Type
    size::NTuple{N, Int}
end
 =#

CoordinateSpace(sz::Int) = CoordinateSpace(Tuple(sz))
CoordinateSpace{0}() = CoordinateSpace(())
CoordinateSpace(sp::CoordinateSpace) = sp

Base.size(sp::CoordinateSpace) = sp.size
Base.size(sp::CoordinateSpace{N}, d::Int) where {N} = d <= N ? size(sp)[d] : 1
Base.axes(A::CoordinateSpace{N}, d::Int) where {N} = d <= N ? axes(A)[d] : Base.OneTo(1)


Base.length(sp::CoordinateSpace) = prod(sp.size)
Base.ndims(::CoordinateSpace{N}) where {N} = N
Base.ndims(::Type{CoordinateSpace{N}}) where {N} = N


Base.in(x::AbstractArray{T, N}, sp::CoordinateSpace{N}) where {T, N} = (size(sp) == size(x))
Base.in(::AbstractArray, ::CoordinateSpace) = false

Base.zeros(sp::CoordinateSpace) = zeros(size(sp))
Base.zeros(::Type{T}, sp::CoordinateSpace) where {T} = zeros(T, size(sp))
Base.ones(sp::CoordinateSpace) = ones(size(sp))
Base.ones(::Type{T}, sp::CoordinateSpace) where {T} = ones(T, size(sp))
Base.rand(sp::CoordinateSpace) = rand(size(sp)...)
Base.rand(::Type{T}, sp::CoordinateSpace) where {T} = rand(T, size(sp)...)
Base.randn(sp::CoordinateSpace) = randn(size(sp))
Base.randn(::Type{T}, sp::CoordinateSpace) where {T} = randn(T, size(sp))

Base.similar(A::AbstractArray, sp::CoordinateSpace) = similar(A, size(sp))
