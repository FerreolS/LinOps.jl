_dims2tuple(dims::Integer) = (Int(dims),)
_dims2tuple(dims::NTuple{N, <:Integer}) where {N} = dims
_dims2tuple(dims::SVector{N, <:Integer}) where {N} = ntuple(i -> Int(dims[i]), Val(N))
_dims2tuple(dims::AbstractVector{<:Integer}) = Tuple(Int.(dims))

function verify_adjoint(A::LinOp)
    x = randn(inputspace(A))
    y = randn(outputspace(A))
    return dot(y, A * x) ≈ dot(A'y, x)
end

"""
    verify_adjoint_composition(H::LinOp)

Check whether `(H' * H) * x` agrees with `H' * (H * x)` and `(H * H') * y`
agrees with `H * (H' * y)` for random `x` and `y` in the input and output
spaces of `H`.
"""
function verify_adjoint_composition(H::LinOp)
    x = randn(inputspace(H))
    y = randn(outputspace(H))
    return ((H' * H) * x ≈ H' * (H * x)) && ((H * H') * y ≈ H * (H' * y))
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
