struct LinOpGrad{I, O, D} <: LinOp{I, O}
    inputspace::I
    outputspace::O
    diff::D
    function LinOpGrad(inputspace::I; diff = nothing) where {I <: AbstractDomain}
        N = ndims(inputspace)
        offsets = _linopgrad_parse_diff(diff, Val(N))
        outputspace = CoordinateSpace((size(inputspace)..., count(>(0), offsets)))
        return new{I, typeof(outputspace), typeof(offsets)}(inputspace, outputspace, offsets)
    end
end

LinOpGrad(sz::NTuple{N, Int}; diff = nothing) where {N} = LinOpGrad(CoordinateSpace(sz); diff = diff)
LinOpGrad(sz::Int; diff = nothing) = LinOpGrad((sz,); diff = diff)

Base.eltype(::LinOpGrad) = Bool

apply_(A::LinOpGrad, x) = compute_grad(x, A.diff)
apply_adjoint_(A::LinOpGrad, x) = compute_grad_adjoint(x, A.diff)

function apply_!(y, A::LinOpGrad, x)
    return compute_grad!(y, x, A.diff)
end

function apply_adjoint_!(y, A::LinOpGrad, x)
    return compute_grad_adjoint!(y, x, A.diff)
end

function compute_grad(x::AbstractArray{T, N}, diff::NTuple{N, Int}) where {T, N}
    backend = get_backend(x)
    Y = KernelAbstractions.zeros(backend, T, size(x)..., count(>(0), diff))
    return compute_grad!(Y, x, diff)
end

function compute_grad_adjoint(x::AbstractArray{T, M}, diff::NTuple{N, Int}) where {T, M, N}
    M >= 1 || throw(ArgumentError("LinOpGrad adjoint input must have at least one dimension"))
    M == N + 1 || throw(ArgumentError("LinOpGrad adjoint input must have one more dimension than diff length"))
    K = count(>(0), diff)
    size(x, M) == K || throw(ArgumentError("LinOpGrad adjoint input must have last dimension equal to $(K), got $(size(x, M))"))
    backend = get_backend(x)
    Y = KernelAbstractions.zeros(backend, T, ntuple(d -> size(x, d), Val(N)))
    return compute_grad_adjoint!(Y, x, diff)
end

function compute_grad!(Y::AbstractArray{T, M}, X::AbstractArray{T, N}, diff::NTuple{N, Int}) where {T, M, N}
    M == N + 1 || throw(ArgumentError("LinOpGrad output must have one more dimension than input"))
    size(Y)[1:N] == size(X) || throw(ArgumentError("LinOpGrad output spatial dimensions must match input size"))
    K = count(>(0), diff)
    size(Y, M) == K || throw(ArgumentError("LinOpGrad output last dimension must be $(K), got $(size(Y, M))"))
    _linopgrad_validate_offsets_for_size(diff, size(X), Val(N))

    fill!(Y, zero(T))
    backend = get_backend(X)
    c = 0
    for d in 1:N
        off = diff[d]
        off == 0 && continue
        c += 1
        idx = _linopgrad_offset_index(Val(N), d, off)
        evt = linopgrad_dif_kernel!(backend)(Y, X, idx, c; ndrange = _linopgrad_ndrange(size(X), idx, Val(N)))
        _linopgrad_wait_or_sync(backend, evt)
    end
    return Y
end

function compute_grad_adjoint!(Y::AbstractArray{T, N}, X::AbstractArray{T, M}, diff::NTuple{N, Int}) where {T, M, N}
    M == N + 1 || throw(ArgumentError("LinOpGrad adjoint input must have one more dimension than output"))
    size(X)[1:N] == size(Y) || throw(ArgumentError("LinOpGrad adjoint spatial dimensions must match output size"))
    K = count(>(0), diff)
    size(X, M) == K || throw(ArgumentError("LinOpGrad adjoint input last dimension must be $(K), got $(size(X, M))"))
    _linopgrad_validate_offsets_for_size(diff, size(Y), Val(N))

    fill!(Y, zero(T))
    backend = get_backend(X)
    c = 0
    for d in 1:N
        off = diff[d]
        off == 0 && continue
        c += 1
        idx = _linopgrad_offset_index(Val(N), d, off)
        evt = linopgrad_dif_adjoint_kernel!(backend)(Y, X, idx, c; ndrange = _linopgrad_ndrange(size(Y), idx, Val(N)))
        _linopgrad_wait_or_sync(backend, evt)
    end
    return Y
end

@inline _linopgrad_offset_index(::Val{N}, d::Int, off::Int) where {N} = CartesianIndex(ntuple(i -> i == d ? off : 0, Val(N)))
@inline _linopgrad_ndrange(sz::NTuple{N, Int}, idx::CartesianIndex{N}, ::Val{N}) where {N} = ntuple(i -> sz[i] - idx[i], Val(N))

function _linopgrad_parse_diff(::Nothing, ::Val{N}) where {N}
    return ntuple(_ -> 1, Val(N))
end

function _linopgrad_parse_diff(diff, ::Val{N}) where {N}
    length(diff) == N || throw(ArgumentError("LinOpGrad diff must have length $(N), got $(length(diff))"))
    offsets = ntuple(i -> Int(diff[i]), Val(N))
    all(>=(0), offsets) || throw(ArgumentError("LinOpGrad diff offsets must be >= 0"))
    return offsets
end

function _linopgrad_validate_offsets_for_size(diff::NTuple{N, Int}, sz::NTuple{N, Int}, ::Val{N}) where {N}
    for d in 1:N
        diff[d] <= sz[d] || throw(ArgumentError("LinOpGrad diff offset along dimension $(d) is $(diff[d]) but size is $(sz[d])"))
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

@kernel function linopgrad_dif_kernel!(Y, X, idx, d)
    I = @index(Global, Cartesian)
    @inbounds Y[I, d] = X[I] - X[I + idx]
end

@kernel function linopgrad_dif_adjoint_kernel!(Y, X, idx, d)
    I = @index(Global, Cartesian)
    @inbounds Y[I] += X[I, d]
    @inbounds Y[I + idx] -= X[I, d]
end
