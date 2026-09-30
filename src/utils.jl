_dims2tuple(dims::Integer) = (Int(dims),)
_dims2tuple(dims::NTuple{N, <:Integer}) where {N} = dims
_dims2tuple(dims::SVector{N, <:Integer}) where {N} = ntuple(i -> Int(dims[i]), Val(N))
_dims2tuple(dims::AbstractVector{<:Integer}) = Tuple(Int.(dims))

function verify_adjoint(A::LinOp)
    x = randn(inputspace(A))
    y = randn(outputspace(A))
    return dot(y, A * x) ≈ dot(A'y, x)
end

@inline function _wait_or_sync(backend, evt)
    if evt === nothing
        applicable(synchronize, backend) && synchronize(backend)
    else
        wait(evt)
    end
    return nothing
end

_atomic_type(::Type{T}) where {T} = T <: Union{Int32, Int64, UInt32, UInt64, Float32, Float64}
_atomic_type(::Type{Complex{T}}) where {T} = _atomic_type(T)

function _backend_supports_atomics(backend)
    KernelAbstractions.supports_atomics(backend) || return false
    if isdefined(KernelAbstractions, :isgpu)
        return backend isa KernelAbstractions.CPU || getfield(KernelAbstractions, :isgpu)(backend)
    end
    return backend isa KernelAbstractions.CPU
end
