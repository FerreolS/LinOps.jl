abstract type AbstractDomain{N} end


Base.size(sp::AbstractDomain) = sp.size
Base.size(sp::AbstractDomain{N}, d::Int) where {N} =
    d < 1 ? throw(ErrorException("arraysize: dimension out of range")) : (d <= N ? size(sp)[d] : 1)
Base.axes(A::AbstractDomain{N}, d::Int) where {N} =
    d < 1 ? throw(BoundsError(axes(A), d)) : (d <= N ? axes(A)[d] : Base.OneTo(1))


Base.length(sp::AbstractDomain) = prod(sp.size)
Base.ndims(::AbstractDomain{N}) where {N} = N
Base.ndims(::Type{<:AbstractDomain{N}}) where {N} = N

Base.eltype(::Type{<:AbstractDomain}) = Bool
Base.eltype(::AbstractDomain) = Bool

Base.in(::AbstractArray, ::AbstractDomain) = false
⊂(::AbstractDomain, ::AbstractDomain) = false
⊂(in::AbstractDomain{N}, sp::AbstractDomain{N}) where {N} = (size(sp) == size(in))

Base.zeros(sp::AbstractDomain) = zeros(size(sp))
Base.ones(sp::AbstractDomain) = ones(size(sp))
Base.rand(sp::AbstractDomain) = rand(size(sp)...)
Base.randn(sp::AbstractDomain) = randn(size(sp))

Base.zeros(::Type{T}, sp::AbstractDomain) where {T} = zeros(T, size(sp))
Base.ones(::Type{T}, sp::AbstractDomain) where {T} = ones(T, size(sp))
Base.rand(::Type{T}, sp::AbstractDomain) where {T} = rand(T, size(sp)...)
Base.randn(::Type{T}, sp::AbstractDomain) where {T} = randn(T, size(sp))

Base.similar(A::AbstractArray, sp::AbstractDomain) = similar(A, size(sp))
Base.similar(A::AbstractArray, ::Type{T}, sp::AbstractDomain) where {T} = similar(A, T, size(sp))

Adapt.adapt_structure(::Any, x::AbstractDomain) = x

struct CoordinateSpace{N} <: AbstractDomain{N}
    size::NTuple{N, Int}
    CoordinateSpace(sz::NTuple{N, Int}) where {N} = new{N}(sz)
end

CoordinateSpace(sz::Int) = CoordinateSpace(Tuple(sz))
CoordinateSpace() = CoordinateSpace(())
CoordinateSpace(sp::CoordinateSpace) = sp

Base.in(x::AbstractArray{T, N}, sp::CoordinateSpace{N}) where {T, N} = (size(sp) == size(x))

struct TypedCoordinateSpace{T, N} <: AbstractDomain{N}
    size::NTuple{N, Int}
    TypedCoordinateSpace(T::Type, sz::NTuple{N, Int}) where {N} = new{T, N}(sz)
end

Base.in(x::AbstractArray{T, N}, sp::TypedCoordinateSpace{T, N}) where {T, N} = (size(sp) == size(x))
⊂(in::TypedCoordinateSpace{T, N}, sp::TypedCoordinateSpace{T, N}) where {T, N} = (size(sp) == size(in))
⊂(in::TypedCoordinateSpace{T1, N}, sp::TypedCoordinateSpace{T2, N}) where {T1, T2, N} = (size(sp) == size(in)) && promote_type(T1, T2) == T2
⊂(in::AbstractDomain{N}, sp::TypedCoordinateSpace{T, N}) where {T, N} = (size(sp) == size(in))
⊂(in::TypedCoordinateSpace{T, N}, sp::AbstractDomain{N}) where {T, N} = (size(sp) == size(in))


TypedCoordinateSpace(T::Type, sz::Int) = TypedCoordinateSpace(T, Tuple(sz))
TypedCoordinateSpace(T::Type) = TypedCoordinateSpace(T, ())
TypedCoordinateSpace(sp::TypedCoordinateSpace) = sp

Base.eltype(::Type{TypedCoordinateSpace{T, N}}) where {T, N} = T
Base.eltype(::TypedCoordinateSpace{T, N}) where {T, N} = T

Base.zeros(sp::TypedCoordinateSpace{T}) where {T} = zeros(T, size(sp))
Base.ones(sp::TypedCoordinateSpace{T}) where {T} = ones(T, size(sp))
Base.rand(sp::TypedCoordinateSpace{T}) where {T} = rand(T, size(sp)...)
Base.randn(sp::TypedCoordinateSpace{T}) where {T} = randn(T, size(sp))

Base.similar(A::AbstractArray, sp::TypedCoordinateSpace{T}) where {T} = similar(A, T, size(sp))


promote_domain(::Type{<:AbstractDomain{N}}, ::Type{<:AbstractDomain{N}}) where {N} = CoordinateSpace{N}
promote_domain(::Type{<:TypedCoordinateSpace{T1, N}}, ::Type{<:TypedCoordinateSpace{T2, N}}) where {T1, T2, N} = TypedCoordinateSpace{promote_type(T1, T2), N}
promote_domain(::Type{<:AbstractDomain{N}}, ::Type{<:TypedCoordinateSpace{T, N}}) where {T, N} = TypedCoordinateSpace{T, N}
promote_domain(::Type{<:TypedCoordinateSpace{T, N}}, ::Type{<:AbstractDomain{N}}) where {T, N} = TypedCoordinateSpace{T, N}
