"""
    LinOpGrad(inputspace; offsets=nothing)
    LinOpGrad(sz; offsets=nothing)

Finite-difference gradient operator.

`offsets` controls per-dimension forward-difference offsets (default is `1` in each
dimension). Nonzero offsets create one output channel per active dimension.

# Examples
```julia
A = LinOpGrad((128, 128))
x = rand(Float32, 128, 128)
g = A * x
```
"""
struct LinOpGrad{I, O, D} <: LinOp{I, O}
    inputspace::I
    outputspace::O
    offsets::D
    function LinOpGrad(inputspace::I; offsets = nothing) where {I <: AbstractDomain}
        N = ndims(inputspace)
        offsets = _linopgrad_parse_offsets(offsets, Val(N))
        outputspace = CoordinateSpace((size(inputspace)..., count(>(0), offsets)))
        _linopgrad_validate_offsets_for_size(offsets, size(inputspace), Val(N))
        return new{I, typeof(outputspace), typeof(offsets)}(inputspace, outputspace, offsets)
    end
    function LinOpGrad(inputspace::I, outputspace::O, offsets::D) where {I <: AbstractDomain, O <: AbstractDomain, D}
        return new{I, O, D}(inputspace, outputspace, offsets)
    end
end

LinOpGrad(sz::NTuple{N, Int}; offsets = nothing) where {N} = LinOpGrad(CoordinateSpace(sz); offsets = offsets)
LinOpGrad(sz::Int; offsets = nothing) = LinOpGrad((sz,); offsets = offsets)


function apply_(A::LinOpGrad, x::AbstractArray{T, N}) where {T, N}
    backend = get_backend(x)
    Y = KernelAbstractions.zeros(backend, T, outputsize(A)...)
    return apply_!(Y, A, x)
end

function apply_adjoint_(A::LinOpGrad, x::AbstractArray{T, M}) where {T, M}
    backend = get_backend(x)
    Y = KernelAbstractions.zeros(backend, T, inputsize(A)...)
    return apply_adjoint_!(Y, A, x)
end

function apply_!(y::AbstractArray{T, M}, (; offsets)::LinOpGrad, x::AbstractArray{T, N}) where {T, M, N}
    fill!(y, zero(T))
    backend = get_backend(x)
    c = 0
    for d in 1:N
        off = offsets[d]
        off == 0 && continue
        c += 1
        idx::CartesianIndex{N} = _linopgrad_offset_index(Val(N), d, off)
        ndrange::NTuple{N, Int} = _linopgrad_ndrange(size(x), idx, Val(N))
        evt = linopgrad_dif_kernel!(backend)(y, x, idx, c; ndrange = ndrange)
        _linopgrad_wait_or_sync(backend, evt)
    end
    return y
end

function apply_adjoint_!(y::AbstractArray{T, N}, A::LinOpGrad, x::AbstractArray{T, M}) where {T, M, N}
    fill!(y, zero(T))
    backend = get_backend(x)
    c = 0
    for d in 1:N
        off = A.offsets[d]
        off == 0 && continue
        c += 1
        idx::CartesianIndex{N} = _linopgrad_offset_index(Val(N), d, off)
        ndrange::NTuple{N, Int} = _linopgrad_ndrange(size(y), idx, Val(N))
        evt = linopgrad_dif_adjoint_kernel!(backend)(y, x, idx, c; ndrange = ndrange)
        _linopgrad_wait_or_sync(backend, evt)
    end
    return y
end


@inline _linopgrad_offset_index(::Val{N}, d::Int, off::Int) where {N} = CartesianIndex(ntuple(i -> i == d ? off : 0, Val(N)))
@inline _linopgrad_ndrange(sz::NTuple{N, Int}, idx::CartesianIndex{N}, ::Val{N}) where {N} = ntuple(i -> sz[i] - Tuple(idx)[i], Val(N))

function _linopgrad_parse_offsets(::Nothing, ::Val{N}) where {N}
    return ntuple(_ -> 1, Val(N))
end

function _linopgrad_parse_offsets(offsets, ::Val{N}) where {N}
    length(offsets) == N || throw(ArgumentError("LinOpGrad offsets must have length $(N), got $(length(offsets))"))
    parsed_offsets = ntuple(i -> Int(offsets[i]), Val(N))
    all(>=(0), parsed_offsets) || throw(ArgumentError("LinOpGrad offsets offsets must be >= 0"))
    return parsed_offsets
end

function _linopgrad_validate_offsets_for_size(offsets::NTuple{N, Int}, sz::NTuple{N, Int}, ::Val{N}) where {N}
    for d in 1:N
        offsets[d] <= sz[d] || throw(ArgumentError("LinOpGrad offsets offset along dimension $(d) is $(offsets[d]) but size is $(sz[d])"))
    end
    return nothing
end

@inline function _linopgrad_wait_or_sync(backend, evt)
    if evt === nothing
        applicable(synchronize, backend) && synchronize(backend)
    else
        wait(evt)
    end
    return nothing
end

@kernel function linopgrad_dif_kernel!(Y, X, idx::CartesianIndex{N}, d::Int) where {N}
    I = @index(Global, Cartesian)
    @inbounds Y[I, d] = X[I] - X[I + idx]
end

@kernel function linopgrad_dif_adjoint_kernel!(Y, X, idx::CartesianIndex{N}, d::Int) where {N}
    I = @index(Global, Cartesian)
    @inbounds Y[I] += X[I, d]
    @inbounds Y[I + idx] -= X[I, d]
end
