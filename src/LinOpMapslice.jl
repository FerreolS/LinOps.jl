struct LinOpMapslice{I, O, P, D} <: LinOp{I, O}
    inputspace::I
    outputspace::O
    operator::P
    dims::D
    function LinOpMapslice(inputspace::I, outputspace::O, operator::P, dims::Union{NTuple{N, Int}, SVector{N, Int}}) where {I, O, P, N}
        dims isa NTuple && (dims = SVector{N, Int}(dims))
        return new{I, O, P, typeof(dims)}(inputspace, outputspace, operator, dims)
    end
end

outputtype(A::LinOpMapslice{I, O, <:LinOp}, x) where {I, O} = outputtype(A.operator, x)
outputtype(A::LinOpAdjoint{O, I, <:LinOpMapslice{I, O, <:LinOp}}, x) where {I, O} = outputtype(adjoint(parent(A).operator), x)

LinOpMapslice(sz::NTuple, operator, dims::NTuple) = LinOpMapslice(sz, operator, collect(dims))
LinOpMapslice(sz::NTuple, operator, dims::Int) = LinOpMapslice(sz, operator, collect((dims,)))

function LinOpMapslice(sz::NTuple, operator::LinOp{I, O}, dims::Vector{Int}) where {I, O}
    maximum(dims) <= length(sz) || throw(ArgumentError("Selected dimensions exceed the number of dimensions in the input space"))
    sz[dims] == inputsize(operator) || throw(ArgumentError("The size of the operator does not match the selected dimensions"))

    if I <: TypedCoordinateSpace
        inputspace = TypedCoordinateSpace(eltype(I), sz)
    else
        inputspace = CoordinateSpace(sz)
    end

    outputsz = (sz[1:(dims[1] - 1)]..., outputsize(operator)..., sz[(dims[end] + 1):length(sz)]...)

    if O <: TypedCoordinateSpace
        outputspace = TypedCoordinateSpace(eltype(O), outputsz)
    else
        outputspace = CoordinateSpace(outputsz)
    end
    return LinOpMapslice(inputspace, outputspace, operator, tuple(dims...))
end

function LinOpMapslice(sz::NTuple{N, Int}, operators::AbstractArray{<:LinOp}, dims::Vector{Int}) where {N}
    maximum(dims) <= length(sz) || throw(ArgumentError("Selected dimensions exceed the number of dimensions in the input space"))
    mapreduce(x -> outputsize(first(operators)) == outputsize(x), &, operators) || throw(ArgumentError("All operators in the array should have the same output size"))
    mapreduce(x -> inputsize(first(operators)) == inputsize(x), &, operators) || throw(ArgumentError("All operators in the array should have the same input size"))
    sz[dims] == inputsize(first(operators)) || throw(ArgumentError("The size of the operator does not match the selected dimensions"))
    remainingdims = trues(N)
    remainingdims[dims] .= false
    sz[remainingdims] == size(operators) || throw(ArgumentError("The number of operators should match the size of the selected dimensions"))

    outputsz = (sz[1:(dims[1] - 1)]..., outputsize(first(operators))..., sz[(dims[end] + 1):length(sz)]...)
    outputspace, inputspace = build_spaces(operators, sz, outputsz)

    return LinOpMapslice(inputspace, outputspace, operators, tuple(dims...))
end


function LinOpMapslice(sz::NTuple{N, Int}, operators::AbstractArray{<:Union{Number, UniformScaling}}, dims::Vector{Int}) where {N}
    maximum(dims) <= length(sz) || throw(ArgumentError("Selected dimensions exceed the number of dimensions in the input space"))
    remainingdims = trues(N)
    remainingdims[dims] .= false
    sz[remainingdims] == size(operators) || throw(ArgumentError("The number of operators should match the size of the selected dimensions"))

    inputspace = CoordinateSpace(sz)
    outputspace = CoordinateSpace(sz)
    return LinOpMapslice(inputspace, outputspace, operators, tuple(dims...))
end

function LinOpMapslice(sz::NTuple{N, Int}, operators::AbstractArray{<:AbstractMatrix}, dims::Vector{Int}) where {N}
    maximum(dims) <= length(sz) || throw(ArgumentError("Selected dimensions exceed the number of dimensions in the input space"))
    length(dims) == 1 || throw(ArgumentError("Only one dimension can be selected for matrix operators"))
    remainingdims = trues(N)
    remainingdims[dims] .= false
    sz[remainingdims] == size(operators) || throw(ArgumentError("The number of operators should match the size of the selected dimensions"))

    sz[dims] == tuple(size(first(operators), 2)) || throw(ArgumentError("The size of the operator does not match the selected dimension"))
    outputsz = (sz[1:(dims[1] - 1)]..., size(first(operators), 1), sz[(dims[1] + 1):length(sz)]...)

    inputspace = CoordinateSpace(sz)
    outputspace = CoordinateSpace(outputsz)
    return LinOpMapslice(inputspace, outputspace, operators, tuple(dims...))
end

Adapt.adapt_structure(to, x::LinOpMapslice{I, O, <:AbstractArray{<:Number}}) where {I, O} = LinOpMapslice(adapt(to, inputspace(x)), adapt(to, outputspace(x)), adapt(to, x.operator), x.dims)
Adapt.adapt_structure(to, x::LinOpMapslice{I, O, <:AbstractArray}) where {I, O} = LinOpMapslice(adapt(to, inputspace(x)), adapt(to, outputspace(x)), adapt.(to, x.operator), x.dims)


function apply_!(y, A::LinOpMapslice{I, O, <:LinOp, SVector{N, Int}}, x) where {N, I, O}
    inputsz = inputsize(A)
    dims = A.dims
    d1 = dims[1]
    ndrange = _mapslice_ndrange(inputsz, d1, N, Val(ndims(I) - N))
    cout = colons(Val(ndims(outputspace(A.operator))))
    cin = colons(Val(ndims(inputspace(A.operator))))
    _LinOpMapslice(y, x, A.operator, cin, cout, d1, ndrange)
    return y
end


function apply_!(y, A::LinOpMapslice{I, O, P, D}, x) where {N, I, O, P <: AbstractArray{<:Union{Number, UniformScaling}}, D <: SVector{N, Int}}
    inputsz = inputsize(A)
    dims = A.dims
    d1 = dims[1]
    ndrange = _mapslice_ndrange(inputsz, d1, N, Val(ndims(I) - N))
    c = colons(Val(N))
    _LinOpMapslice(y, x, A.operator, c, c, d1, ndrange)
    return y
end


function apply_!(y, A::LinOpMapslice{I, O, P, D}, x) where {N, I, O, P <: AbstractArray, D <: SVector{N, Int}}
    inputsz = inputsize(A)
    dims = A.dims
    d1 = dims[1]
    ndrange = _mapslice_ndrange(inputsz, d1, N, Val(ndims(I) - N))
    cout = colons(Val(ndims(outputspace(first(A.operator)))))
    cin = colons(Val(ndims(inputspace(first(A.operator)))))
    _LinOpMapslice(y, x, A.operator, cin, cout, d1, ndrange)
    return y
end


@inline _mapslice_ndrange(outputsz, d1::Int, M::Int, ::Val{K}) where {K} = ntuple(i -> i < d1 ? outputsz[i] : outputsz[i + M], Val(K))

function apply_adjoint_!(y, A::LinOpMapslice{I, O, P, D}, x) where {N, I, O, P <: LinOp, D <: SVector{N, Int}}
    outputsz = outputsize(A)
    d1 = A.dims[1]
    M = ndims(outputspace(A.operator))
    ndrange = _mapslice_ndrange(outputsz, d1, M, Val(ndims(I) - N))
    cin = colons(Val(ndims(outputspace(A.operator))))
    cout = colons(Val(ndims(inputspace(A.operator))))
    _LinOpMapslice(y, x, A.operator', cin, cout, d1, ndrange)
    return y
end


function apply_adjoint_!(y, A::LinOpMapslice{I, O, P, D}, x) where {N, I, O, P <: AbstractArray{<:Union{Number, UniformScaling}}, D <: SVector{N, Int}}
    outputsz = outputsize(A)
    d1 = A.dims[1]
    ndrange = _mapslice_ndrange(outputsz, d1, N, Val(ndims(I) - N))
    c = colons(Val(N))
    _LinOpMapslice(y, x, map(adjoint, A.operator), c, c, d1, ndrange)
    return y
end


function apply_adjoint_!(y, A::LinOpMapslice{I, O, P, D}, x) where {N, I, O, P <: AbstractArray, D <: SVector{N, Int}}
    outputsz = outputsize(A)
    d1 = A.dims[1]
    M = ndims(outputspace(first(A.operator)))
    ndrange = _mapslice_ndrange(outputsz, d1, M, Val(ndims(I) - N))
    cout = colons(Val(ndims(outputspace(first(A.operator)))))
    cin = colons(Val(ndims(inputspace(first(A.operator)))))
    _LinOpMapslice(y, x, map(adjoint, A.operator), cin, cout, d1, ndrange)
    return y
end

function _LinOpMapslice(Y, X, A, cin, cout, dims, ndrange)
    for I in CartesianIndices(ndrange)
        I1 = CartesianIndex(I.I[1:(dims - 1)])
        I2 = CartesianIndex(I.I[dims:end])
        #view(Y, I1, colons(Val(ndims(outputspace(A))))..., I2) .= A * view(X, I1, colons(Val(ndims(inputspace(A))))..., I2)
        mul!(view(Y, I1, cout..., I2), A, view(X, I1, cin..., I2))
        #mul!(Y[I1, cout..., I2], A, X[I1, cin..., I2])
    end
    #Y[I1, .., I2] .= A * X[I1, .., I2]
    return
end

function _LinOpMapslice(Y, X, A::AbstractArray, cin, cout, dims, ndrange)
    for I in CartesianIndices(ndrange)
        I1 = CartesianIndex(I.I[1:(dims - 1)])
        I2 = CartesianIndex(I.I[dims:end])
        #view(Y, I1, colons(Val(ndims(outputspace(A))))..., I2) .= A * view(X, I1, colons(Val(ndims(inputspace(A))))..., I2)
        mul!(view(Y, I1, cout..., I2), A[I], view(X, I1, cin..., I2))
    end

    return
end

#=

@kernel function LinOpMapslice_kernel(Y, X, A, cin, cout, dims)
    I = @index(Global, Cartesian)
    I1 = CartesianIndex(I.I[1:(dims - 1)])
    I2 = CartesianIndex(I.I[dims:end])
    #view(Y, I1, colons(Val(ndims(outputspace(A))))..., I2) .= A * view(X, I1, colons(Val(ndims(inputspace(A))))..., I2)
    mul!(view(Y, I1, cout..., I2), A, view(X, I1, cin..., I2))

    #Y[I1, .., I2] .= A * X[I1, .., I2]
end

@kernel function LinOpMapslice_kernel(Y, X, A::AbstractArray, cin, cout, dims)
    I = @index(Global, Cartesian)
    I1 = CartesianIndex(I.I[1:(dims - 1)])
    I2 = CartesianIndex(I.I[dims:end])
    #view(Y, I1, colons(Val(ndims(outputspace(A))))..., I2) .= A * view(X, I1, colons(Val(ndims(inputspace(A))))..., I2)
    mul!(view(Y, I1, cout..., I2), A[I], view(X, I1, cin..., I2))

end

@kernel function LinOpMapslice_kernel(Y, X, A,cin, cout)
     I = @index(Global, Cartesian)
    #view(Y, I1, colons(Val(ndims(outputspace(A))))..., I2) .= A * view(X, I1, colons(Val(ndims(inputspace(A))))..., I2)
   #@show I
   mul!(view(Y,  I,cout...), A, view(X, I,cin...))

    #Y[I1, .., I2] .= A * X[I1, .., I2]
end =#

function build_spaces(operators, inputsz, outputsz)
    outputdomain = mapreduce(x -> typeof(outputspace(x)), promote_domain, operators)
    inputdomain = mapreduce(x -> typeof(inputspace(x)), promote_domain, operators)

    if inputdomain === TypedCoordinateSpace
        ginputspace = TypedCoordinateSpace(eltype(inputdomain), inputsz)
    else
        ginputspace = CoordinateSpace(inputsz)
    end
    if outputdomain === TypedCoordinateSpace
        goutputspace = TypedCoordinateSpace(eltype(outputdomain), outputsz)
    else
        goutputspace = CoordinateSpace(outputsz)
    end
    return goutputspace, ginputspace
end
