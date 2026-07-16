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


@inline Base.length(sp::AbstractDomain) = prod(sp.size)
@inline Base.ndims(::AbstractDomain{N}) where {N} = N
@inline Base.ndims(::Type{<:AbstractDomain{N}}) where {N} = N

Base.eltype(::Type{<:AbstractDomain}) = Bool
Base.eltype(::AbstractDomain) = Bool

Base.in(::AbstractArray, ::AbstractDomain) = false
"""
    ⊂(a, b)

Domain inclusion predicate. Returns `true` when domain `a` is compatible with `b`.
"""
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

"""
    CoordinateSpace

Shape-only domain descriptor.

Use `CoordinateSpace((n1, n2, ...))` when only array dimensions matter,
or `CoordinateSpace(T, (n1, n2, ...))` to constrain element type.
"""
struct CoordinateSpace{T, N, A} <: AbstractDomain{N}
    size::NTuple{N, Int}
    function CoordinateSpace(T::Type, sz::NTuple{N, Int}, A::Type) where {N}
        T <: Number || throw(ArgumentError("CoordinateSpace element type must be a subtype of Number"))
        A <: AbstractArray || throw(ArgumentError("CoordinateSpace array type must be a subtype of AbstractArray"))
        return new{T, N, A}(sz)
    end
end

CoordinateSpace{T, N, A}(sz::NTuple{N, Int}) where {T, N, A} = CoordinateSpace(T, sz, A)

CoordinateSpace() = CoordinateSpace(Number, ())
CoordinateSpace(sz::Int) = CoordinateSpace(Number, Tuple(sz))
CoordinateSpace(sz::NTuple{N, Int}) where {N} = CoordinateSpace(Number, sz)
CoordinateSpace(T::Type, sz::NTuple{N, Int}) where {N} = CoordinateSpace(T, sz, AbstractArray)
CoordinateSpace(sp::CoordinateSpace) = sp


Base.eltype(::Type{<:CoordinateSpace{T, N}}) where {T, N} = isconcretetype(T) ? T : Bool
Base.eltype(::CoordinateSpace{T, N}) where {T, N} = isconcretetype(T) ? T : Bool


@inline get_type(::Type{<:CoordinateSpace{T, N}}) where {T, N} = T
@inline get_type(::CoordinateSpace{T, N}) where {T, N} = T


@inline get_storage(::CoordinateSpace{T, N, A}) where {T, N, A} = A
@inline get_storage(::Type{CoordinateSpace{T, N, A}}) where {T, N, A} = A

@inline function Base.in(x::AbstractArray{T1, N}, sp::CoordinateSpace{T2, N, A}) where {T1, N, T2, A}
    storagecheck = x isa SubArray ? (parent(x) isa A) : (x isa A)
    return (size(sp) == size(x)) && storagecheck && ((T1 <: T2) ||  (promote_type(T1, T2) <: T2))
end

⊂(in::CoordinateSpace{T, N, A}, sp::CoordinateSpace{T, N, A}) where {T, N, A} = (size(sp) == size(in))
⊂(in::CoordinateSpace{T1, N, A}, sp::CoordinateSpace{T2, N, A}) where {N, T1, T2, A} = (size(sp) == size(in)) && (!isconcretetype(T1) || !isconcretetype(T2) || promote_type(T1, T2) == T2)
⊂(in::CoordinateSpace{T1, N, A1}, sp::CoordinateSpace{T2, N, A2}) where {N, T1, T2, A1, A2} = (size(sp) == size(in)) && (!isconcretetype(T1) || !isconcretetype(T2) || promote_type(T1, T2) == T2)&& A1 <: A2
⊂(in::CoordinateSpace{T1, N, AbstractArray}, sp::CoordinateSpace{T2, N}) where {N, T1, T2} = (size(sp) == size(in)) && (!isconcretetype(T1) || !isconcretetype(T2) || promote_type(T1, T2) == T2)

@inline function Base.zeros(sp::CoordinateSpace{T, N, A}) where {T, N, A}
    To = isconcretetype(T) ? T : Float64
    Ao = isconcretetype(A{To, N}) ? A : Array
    return fill!(Ao{To, N}(undef, size(sp)), zero(To))
end
@inline function Base.ones(sp::CoordinateSpace{T, N, A}) where {T, N, A}
    To = isconcretetype(T) ? T : Float64
    Ao = isconcretetype(A{To, N}) ? A : Array
    return fill!(Ao{To, N}(undef, size(sp)), one(To))
end

Base.rand(sp::CoordinateSpace{T, N, <:Union{Array, AbstractArray}}) where {T, N} = isconcretetype(T) ? rand(T, size(sp)) : rand(Float64, size(sp))
Base.randn(sp::CoordinateSpace{T, N, <:Union{Array, AbstractArray}}) where {T, N} = isconcretetype(T) ? randn(T, size(sp)) : randn(Float64, size(sp))
Base.similar(t::Type, sp::CoordinateSpace{T, N, <:Union{Array, AbstractArray}}) where {T, N} = similar(t, size(sp))
@inline Base.similar(A::AbstractArray, sp::CoordinateSpace{T, N}) where {T, N} = isconcretetype(T) ? similar(A, T, size(sp)) : similar(A, size(sp))
@inline function Base.similar(sp::CoordinateSpace{T, N, A}) where {T, N, A}
    To = isconcretetype(T) ? T : Float64
    Ao = isconcretetype(A{To, N}) ? A : Array
    return similar(Ao{To, N}, size(sp))
end

@inline function Adapt.adapt_structure(to::Any, x::CoordinateSpace{T, N, A}) where {T, N, A}
    To = eltype(to)
    To = isconcretetype(To) ? To : T
    Ao = parameterless(to)
    return CoordinateSpace(To, x.size, Ao)
end

"""
    promote_domain(A, B)

Return a domain type able to represent values compatible with domaiqns `A` and `B`.
"""
promote_domain(::Type{<:AbstractDomain{N}}, ::Type{<:AbstractDomain{N}}) where {N} = CoordinateSpace{Number, N, AbstractArray}
promote_domain(::Type{CoordinateSpace{T1, N, AbstractArray}}, ::Type{CoordinateSpace{T2, N, AbstractArray}}) where {T1, N, T2} = CoordinateSpace{promote_type(T1, T2), N, AbstractArray}
promote_domain(::Type{CoordinateSpace{T1, N, A}}, ::Type{CoordinateSpace{T2, N, AbstractArray}}) where {T1, N, T2, A} = CoordinateSpace{promote_type(T1, T2), N, A}
promote_domain(::Type{CoordinateSpace{T1, N, AbstractArray}}, ::Type{CoordinateSpace{T2, N, A}}) where {T1, N, T2, A} = CoordinateSpace{promote_type(T1, T2), N, A}
