"""
    AbstractDomain{N}

Abstract supertype for domain descriptors of dimension `N`.

Domains encode shape (and optionally element type) constraints used by `LinOp` objects.
"""
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
"""
    ⊂(a, b)

Domain inclusion predicate. Returns `true` when domain `a` is compatible with `b`.
"""
⊂(::AbstractDomain, ::AbstractDomain) = false
⊂(in::AbstractDomain{N}, sp::AbstractDomain{N}) where {N} = (size(sp) == size(in))

Adapt.adapt_structure(::Any, x::AbstractDomain) = x

"""
    CoordinateSpace

Shape-only domain descriptor.

Use `CoordinateSpace((n1, n2, ...))` when only array dimensions matter.
"""
struct CoordinateSpace{N} <: AbstractDomain{N}
    size::NTuple{N, Int}
    CoordinateSpace(sz::NTuple{N, Int}) where {N} = new{N}(sz)
end

CoordinateSpace(sz::Int) = CoordinateSpace(Tuple(sz))
CoordinateSpace() = CoordinateSpace(())
CoordinateSpace(sp::CoordinateSpace) = sp

Base.zeros(sp::CoordinateSpace) = zeros(size(sp))
Base.ones(sp::CoordinateSpace) = ones(size(sp))
Base.rand(sp::CoordinateSpace) = rand(size(sp)...)
Base.randn(sp::CoordinateSpace) = randn(size(sp))

Base.zeros(::Type{T}, sp::CoordinateSpace) where {T} = zeros(T, size(sp))
Base.ones(::Type{T}, sp::CoordinateSpace) where {T} = ones(T, size(sp))
Base.rand(::Type{T}, sp::CoordinateSpace) where {T} = rand(T, size(sp)...)
Base.randn(::Type{T}, sp::CoordinateSpace) where {T} = randn(T, size(sp))

Base.similar(A::AbstractArray, sp::CoordinateSpace) = similar(A, size(sp))
Base.similar(A::AbstractArray, ::Type{T}, sp::CoordinateSpace) where {T} = similar(A, T, size(sp))

Base.in(x::AbstractArray{T, N}, sp::CoordinateSpace{N}) where {T, N} = (size(sp) == size(x))

"""
    TypedCoordinateSpace{T,N}

Domain descriptor carrying both shape and element type `T`.
"""
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

Adapt.adapt_structure(_, x::TypedCoordinateSpace) = x
function Adapt.adapt_structure(::Type{<:AbstractArray{T}}, x::TypedCoordinateSpace{Tx}) where {T, Tx}
    Tx <: Complex && T <: Real && return TypedCoordinateSpace(Complex{T}, size(x))
    return TypedCoordinateSpace(T, size(x))
end


Base.eltype(::Type{TypedCoordinateSpace{T, N}}) where {T, N} = T
Base.eltype(::TypedCoordinateSpace{T, N}) where {T, N} = T

Base.zeros(sp::TypedCoordinateSpace{T}) where {T} = zeros(T, size(sp))
Base.ones(sp::TypedCoordinateSpace{T}) where {T} = ones(T, size(sp))
Base.rand(sp::TypedCoordinateSpace{T}) where {T} = rand(T, size(sp)...)
Base.randn(sp::TypedCoordinateSpace{T}) where {T} = randn(T, size(sp))

Base.similar(A::AbstractArray, sp::TypedCoordinateSpace{T}) where {T} = similar(A, T, size(sp))


"""
    DeviceTypedCoordinateSpace{T,N,B}

Domain descriptor carrying shape, element type `T`, and backend type `B`.
"""
struct StoredCoordinateSpace{T, N, S} <: AbstractDomain{N}
    size::NTuple{N, Int}
    function StoredCoordinateSpace(::Type{T}, sz::NTuple{N, Int}, storage) where {T, N}
        storage <: AbstractArray{T, N} || throw(ArgumentError("storage must be an AbstractArray of type $T and dimension $N"))
        return new{T, N, storage}(sz)
    end
end

StoredCoordinateSpace(T::Type, sz::Int, storage) = StoredCoordinateSpace(T, Tuple(sz), storage)
StoredCoordinateSpace(sz::Int) = StoredCoordinateSpace(Tuple(sz))
StoredCoordinateSpace(sz::NTuple{N, Int}) where {N} = StoredCoordinateSpace(Tuple(sz), Array{Float64, N})
StoredCoordinateSpace(T::Type, sz::NTuple{N, Int}) where {N} = StoredCoordinateSpace(T, Tuple(sz), Array{T, N})
StoredCoordinateSpace(sz::NTuple{N, Int}, storage) where {N} = StoredCoordinateSpace(eltype(storage), sz, storage)
StoredCoordinateSpace(sp::StoredCoordinateSpace) = sp


Base.in(x::AbstractArray{T, N}, sp::StoredCoordinateSpace{T, N}) where {T, N} = (size(sp) == size(x)) && (x isa get_storage(sp))
⊂(in::StoredCoordinateSpace{T, N}, sp::StoredCoordinateSpace{T, N}) where {T, N} = (size(sp) == size(in)) && (get_storage(in) === get_storage(sp))
⊂(in::StoredCoordinateSpace{T1, N}, sp::StoredCoordinateSpace{T2, N}) where {T1, T2, N} = (size(sp) == size(in)) && promote_type(T1, T2) == T2 && (get_storage(in) === get_storage(sp))
⊂(in::StoredCoordinateSpace{T, N}, sp::AbstractDomain{N}) where {T, N} = (size(sp) == size(in))
⊂(in::StoredCoordinateSpace{T, N}, sp::TypedCoordinateSpace{T, N}) where {T, N} = (size(sp) == size(in))
⊂(in::StoredCoordinateSpace{T1, N}, sp::TypedCoordinateSpace{T2, N}) where {T1, T2, N} = (size(sp) == size(in)) && promote_type(T1, T2) == T2

Adapt.adapt_structure(_, x::StoredCoordinateSpace) = x
function Adapt.adapt_structure(to::Type{<:AbstractArray{T}}, x::StoredCoordinateSpace{Tx, N}) where {T, Tx, N}
    Tx <: Complex && T <: Real && return StoredCoordinateSpace(size(x), parameterless(to){Complex{T}, N})
    T === Any && return StoredCoordinateSpace(Tx, size(x), parameterless(to){Tx, N})
    return StoredCoordinateSpace(T, size(x), parameterless(to){T, N})
end

Base.eltype(::Type{StoredCoordinateSpace{T, N}}) where {T, N} = T
Base.eltype(::StoredCoordinateSpace{T, N}) where {T, N} = T
Base.ndims(::StoredCoordinateSpace{T, N}) where {T, N} = N
Base.ndims(::Type{<:StoredCoordinateSpace{T, N}}) where {T, N} = N

get_storage(::StoredCoordinateSpace{T, N, S}) where {T, N, S} = S

Base.zeros(sp::StoredCoordinateSpace{T}) where {T} = fill!(get_storage(sp)(undef, size(sp)), zero(T))
Base.ones(sp::StoredCoordinateSpace{T}) where {T} = fill!(get_storage(sp)(undef, size(sp)), one(T))
Base.rand(sp::StoredCoordinateSpace{T, N, S}) where {T, N, S <: Array} = rand(T, size(sp)...)
Base.randn(sp::StoredCoordinateSpace{T, N, S}) where {T, N, S <: Array} = randn(T, size(sp))

Base.similar(A::AbstractArray, sp::StoredCoordinateSpace{T}) where {T} = similar(A, T, size(sp))
Base.similar(sp::StoredCoordinateSpace{T}) where {T} = similar(get_storage(sp), size(sp))


"""
    promote_domain(A, B)

Return a domain type able to represent values compatible with domains `A` and `B`.
"""
promote_domain(::Type{<:AbstractDomain{N}}, ::Type{<:AbstractDomain{N}}) where {N} = CoordinateSpace{N}
promote_domain(::Type{<:TypedCoordinateSpace{T1, N}}, ::Type{<:TypedCoordinateSpace{T2, N}}) where {T1, T2, N} = TypedCoordinateSpace{promote_type(T1, T2), N}
promote_domain(::Type{<:StoredCoordinateSpace{T1, N}}, ::Type{<:StoredCoordinateSpace{T2, N}}) where {T1, T2, N} = StoredCoordinateSpace{promote_type(T1, T2), N}
promote_domain(::Type{<:AbstractDomain{N}}, ::Type{<:TypedCoordinateSpace{T, N}}) where {T, N} = TypedCoordinateSpace{T, N}
promote_domain(::Type{<:TypedCoordinateSpace{T, N}}, ::Type{<:AbstractDomain{N}}) where {T, N} = TypedCoordinateSpace{T, N}
promote_domain(::Type{<:AbstractDomain{N}}, ::Type{<:StoredCoordinateSpace{T, N}}) where {T, N} = StoredCoordinateSpace{T, N}
promote_domain(::Type{<:StoredCoordinateSpace{T, N}}, ::Type{<:AbstractDomain{N}}) where {T, N} = StoredCoordinateSpace{T, N}
promote_domain(::Type{<:TypedCoordinateSpace{T1, N}}, ::Type{<:StoredCoordinateSpace{T2, N}}) where {T1, T2, N} = StoredCoordinateSpace{promote_type(T1, T2), N}
promote_domain(::Type{<:StoredCoordinateSpace{T1, N}}, ::Type{<:TypedCoordinateSpace{T2, N}}) where {T1, T2, N} = StoredCoordinateSpace{promote_type(T1, T2), N}
