_dims2tuple(dims::Integer) = (Int(dims),)
_dims2tuple(dims::NTuple{N, <:Integer}) where {N} = dims
_dims2tuple(dims::SVector{N, <:Integer}) where {N} = ntuple(i -> Int(dims[i]), Val(N))
_dims2tuple(dims::AbstractVector{<:Integer}) = Tuple(Int.(dims))
