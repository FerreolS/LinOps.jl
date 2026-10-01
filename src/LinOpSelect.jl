"""
    LinOpSelect(sz, indices)
    LinOpSelect(sz, ranges)
    LinOpSelect(mask::AbstractArray{Bool})
    LinOpSelect(inputspace, outputspace, indices)

Selection operator that gathers indexed entries from its input.

The shape-and-index constructor accepts a vector of linear integer indices or
`CartesianIndex` values. It preserves their order and allows repeated indices.
The range-tuple constructor accepts one range per input dimension and preserves
the shape of the selected subarray.
The boolean-mask constructor selects the `true` entries in linear order. The
explicit-domain constructor accepts the input and output domains directly; the
number of indices must equal the number of elements in the output domain.

The adjoint scatters values back into the input domain, adding contributions
when an index occurs more than once.

# Examples
```julia
A = LinOpSelect((3,), [2, 2])
A * [10, 20, 30]  # [20, 20]
A' * [1, 2]       # [0, 3, 0]

M = LinOpSelect([false, true, true])
M * [10, 20, 30]  # [20, 30]

X = reshape(1:20, 4, 5)
S = LinOpSelect(size(X), (2:3, 2:4))
S * X  # X[2:3, 2:4]
```
"""
struct LinOpSelect{I, O, D} <: LinOp{I, O}
    inputspace::I
    outputspace::O
    index::D
    function LinOpSelect(inputspace::I, outputspace::O, list::D) where {D, I <: AbstractDomain, O <: AbstractDomain}
        length(list) == length(outputspace) || throw(ArgumentError("Index list length must match the number of elements in the output space"))
        linear_index = _linopselect_linear_indices(inputspace, list)
        return new{I, O, typeof(linear_index)}(inputspace, outputspace, linear_index)
    end
end

_linopselect_linear_indices(inputspace, index::AbstractArray{<:CartesianIndex}) = LinearIndices(size(inputspace))[index]
_linopselect_linear_indices(inputspace, index::AbstractArray{<:Integer}) = index

LinOpSelect(sz::NTuple{N, Int}, list::AbstractVector) where {N} = LinOpSelect(LinOps.CoordinateSpace(sz), LinOps.CoordinateSpace(length(list)), list)

function LinOpSelect(sz::NTuple{N, Int}, ranges::Tuple) where {N}
    length(ranges) == N || throw(ArgumentError("One range is required for each input dimension"))
    all(range -> range isa AbstractRange, ranges) || throw(ArgumentError("Selection indices must be ranges"))

    output_size = map(length, ranges)
    linear_indices = LinearIndices(sz)
    index = [linear_indices[I] for I in CartesianIndices(ranges)]
    return LinOpSelect(LinOps.CoordinateSpace(sz), LinOps.CoordinateSpace(output_size), index)
end

function LinOpSelect(selected::AbstractArray{Bool, N}) where {N}
    sz = size(selected)
    list = findall(selected)
    return LinOpSelect(sz, list)
end


#apply_((; index)::LinOpSelect, x) = x[index] #view(x,A.index)

function apply_(A::LinOpSelect, x)
    backend = get_backend(x)
    Y = KernelAbstractions.zeros(backend, eltype(x), outputsize(A)...)
    return apply_!(Y, A, x)
end


function apply_!(y, A::LinOpSelect, x)
    backend = get_backend(x)
    index = _linopselect_device_indices(A, backend)
    evt = linopselect_gather_kernel!(backend)(y, x, index; ndrange = length(index))
    _wait_or_sync(backend, evt)
    return y
end

function apply_adjoint_(A::LinOpSelect, x)
    backend = get_backend(x)
    y = KernelAbstractions.allocate(backend, eltype(x), inputsize(A))
    apply_adjoint_!(y, A, x)
    return y
end

function apply_adjoint_!(y, A::LinOpSelect, x)
    backend = get_backend(x)
    ChainRulesCore.@ignore_derivatives begin
        fill!(y, zero(eltype(y)))
        device_index = _linopselect_device_indices(A, backend)
        if _backend_supports_atomics(backend) && _atomic_type(eltype(x))
            _linopselect_atomic_scatter_add!(backend, y, x, device_index, length(A.index))
        else
            evt = linopselect_reduce_scatter_kernel!(backend)(y, x, device_index; ndrange = length(y))
            _wait_or_sync(backend, evt)
        end
    end
    return y
end

function _linopselect_atomic_scatter_add!(backend, y, x::AbstractArray{<:Complex}, device_index, nindices)
    real_type = typeof(real(zero(eltype(x))))
    y_flat = vec(y)
    real_values = KernelAbstractions.zeros(backend, real_type, (length(y_flat),))
    imag_values = KernelAbstractions.zeros(backend, real_type, (length(y_flat),))
    evt = linopselect_complex_scatter_add_kernel!(backend)(real_values, imag_values, x, device_index; ndrange = nindices)
    _wait_or_sync(backend, evt)
    evt = linopselect_complex_combine_kernel!(backend)(y_flat, real_values, imag_values; ndrange = length(y_flat))
    _wait_or_sync(backend, evt)
    return y
end

function _linopselect_atomic_scatter_add!(backend, y, x, device_index, nindices)
    evt = linopselect_scatter_add_kernel!(backend)(vec(y), x, device_index; ndrange = nindices)
    _wait_or_sync(backend, evt)
    return y
end


function _linopselect_device_indices(A, backend)
    (backend === get_backend(A.index)) && return A.index
    device_index = KernelAbstractions.allocate(backend, Int, size(A.index))
    copyto!(device_index, A.index)
    return device_index
end


@kernel function linopselect_scatter_add_kernel!(Y, X, index)
    j = @index(Global, Linear)
    @inbounds KernelAbstractions.@atomic Y[index[j]] += X[j]
end

@kernel function linopselect_gather_kernel!(Y, X, index)
    j = @index(Global, Linear)
    @inbounds Y[j] = X[index[j]]
end

@kernel function linopselect_complex_scatter_add_kernel!(Yreal, Yimag, X, index)
    j = @index(Global, Linear)
    @inbounds KernelAbstractions.@atomic Yreal[index[j]] += real(X[j])
    @inbounds KernelAbstractions.@atomic Yimag[index[j]] += imag(X[j])
end

@kernel function linopselect_complex_combine_kernel!(Y, Yreal, Yimag)
    i = @index(Global, Linear)
    @inbounds Y[i] = complex(Yreal[i], Yimag[i])
end

@kernel function linopselect_reduce_scatter_kernel!(Y, X, index)
    i = @index(Global, Linear)
    value = zero(eltype(Y))
    @inbounds for j in eachindex(index)
        if index[j] == i
            value += X[j]
        end
    end
    @inbounds    Y[i] = value
end

function Base.:*(left::D, right::LinOpAdjoint{O, I, D}) where {I, O, D <: LinOpSelect}
    if parent(right) === left && allunique(left.index)
        return UniformScaling(1)
    end
    return LinOpCompose(left, right)
end

function Base.:*(left::LinOpAdjoint{O, I, D}, right::D) where {I, O, D <: LinOpSelect}
    if parent(left) === right
        diag = zeros(inputspace(right))
        for index in right.index
            diag[index] += one(eltype(diag))
        end
        return LinOpDiag(inputspace(right), diag)
    end
    return LinOpCompose(left, right)
end
