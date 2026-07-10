"""
NonuniformFFTs extension for LinOps optional NFFT operators.

This module activates `has_operator(:nfft)` and provides NonuniformFFTs-backed
`LinOpNFFT` constructors and adaptation methods.
"""
module LinOpsNonuniformFFTsExt
import LinOps: LinOpNFFT, CoordinateSpace, _dims2tuple
import NonuniformFFTs: PlanNUFFT, exec_type1!, exec_type2!, set_points!

using Adapt
using KernelAbstractions
using LinOps
using NonuniformFFTs
using TypeUtils

LinOps.has_operator(::Val{:nfft}) = true
LinOps.operator_backend(::Val{:nfft}) = :nonuniformffts

"""
    LinOpNFFT(::Type{T}, sz, points; kwargs...)

Create a NonuniformFFTs-backed NFFT operator using sample `points` and grid shape `sz`.
"""
# Real-to-complex FFT.
function LinOpNFFT(
        ::Type{T},
        sz::NTuple{N, Int},
        points::NTuple{M, AbstractVector{T2}};
        dims = :,
        kwargs...
    ) where {T1 <: Real, T <: Union{T1, Complex{T1}}, T2, N, M}

    if T1 != T2
        points = map(p -> convert.(T1, p), points)
    end

    backend = get_backend(points[1])

    if dims isa Colon
        plan_nufft = PlanNUFFT(T, sz; backend = backend, kwargs...)
        outputspace = CoordinateSpace(Complex{T1}, size(plan_nufft), Array)
        inputspace = CoordinateSpace(T, (length(points[1]),), Array)
    else
        dims = _dims2tuple(dims)
        ndd = length(dims)
        if dims == tuple((1:length(dims))...)
            ntrans = prod(sz[(ndd + 1):end])
            plan_nufft = PlanNUFFT(T, sz[1:ndd]; backend = backend, ntransforms = Val(ntrans), kwargs...)
        else
            throw(ArgumentError("Unsupported dims argument: $dims, only Colon or first dimensions supported"))
        end
        outputspace = CoordinateSpace(Complex{T1}, tuple(size(plan_nufft)..., sz[(ndd + 1):end]...), Array)
        inputspace = CoordinateSpace(T, (length(points[1]), sz[(ndd + 1):end]...), Array)
    end
    set_points!(plan_nufft, points)

    return LinOpNFFT(inputspace, outputspace, plan_nufft, dims, sz, points)
end

function Base.show(io::IO, ::MIME"text/plain", A::LinOpNFFT)
    print(io, "Linear Operator: ")
    println(io, summary(A))
    show(io, A.plan)
    return
end

function LinOps.apply_!(y, A::LinOpNFFT{I, O, <:PlanNUFFT{T, N, M}, Colon}, x) where {T, N, M, I, O}
    return exec_type1!(y, A.plan, x)
end

function LinOps.apply_!(y, A::LinOpNFFT{I, O, <:PlanNUFFT{T, N, M}}, x) where {T, N, M, I, O}
    ndd = length(A.dims)
    szin = inputsize(A)
    szout = outputsize(A)
    outer = szin[2:end]
    innerin = szin[1]
    innerout = szout[1:ndd]

    ntrans = prod(outer)
    _x = reshape(x, innerin..., :)
    __x = ntuple(i -> view(_x, :, i), ntrans)
    _y = reshape(y, innerout..., :)
    __y = ntuple(i -> view(_y, ntuple(_ -> Colon(), ndd)..., i), ntrans)

    exec_type1!(__y, A.plan, __x)
    return y
end

function LinOps.apply_adjoint_!(y, A::LinOpNFFT{I, O, <:PlanNUFFT{T, N, M}, Colon}, x) where {T, N, M, I, O}
    exec_type2!(y, A.plan, x)  # returns Tuple{output...}; discard and return y directly
    return y
end

function LinOps.apply_adjoint_!(y, A::LinOpNFFT{I, O, <:PlanNUFFT{T, N, M}}, x) where {T, N, M, I, O}
    ndd = length(A.dims)
    szin = inputsize(A)
    szout = outputsize(A)
    outer = szin[2:end]
    innerin = szin[1]
    innerout = szout[1:ndd]

    ntrans = prod(outer)
    _y = reshape(y, innerin..., :)
    __y = ntuple(i -> view(_y, :, i), ntrans)
    _x = reshape(x, innerout..., :)
    __x = ntuple(i -> view(_x, ntuple(_ -> Colon(), ndd)..., i), ntrans)


    exec_type2!(__y, A.plan, __x)
    return y
end

function Adapt.adapt_structure(to, x::LinOpNFFT)

    if eltype(to) === Any
        T = eltype(inputspace(x))
        tmp = to{T}(undef, 0)
    else
        T = eltype(to)
        tmp = to(undef, 0)
    end
    backend = get_backend(tmp)
    points = adapt(to, x.points)
    return LinOpNFFT(T, x.size, points; dims = x.dims, backend = backend)  # construct new operator with adapted type and same points
end

end
