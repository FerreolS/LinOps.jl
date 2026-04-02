module LinOpsNonuniformFFTsExt
using LinOps, NonuniformFFTs
import LinOps: TypedCoordinateSpace, LinOpAdjoint, LinOpNFFT, apply_!, apply_adjoint_!, outputtype, inputtype
import NonuniformFFTs: PlanNUFFT, exec_type1!, exec_type2!, set_points!

# Real-to-complex FFT.
function LinOpNFFT(
        ::Type{T},
        sz::NTuple{N, Int},
        points::NTuple{M, AbstractVector{T2}};
        kwargs...
    ) where {T1 <: Real, T <: Union{T1, Complex{T1}}, T2, N, M}

    if T1 != T2
        points = map(p -> convert.(T1, p), points)
    end
    plan_nufft = PlanNUFFT(T, sz; kwargs...)
    set_points!(plan_nufft, points)
    outputspace = TypedCoordinateSpace(Complex{T1}, size(plan_nufft))
    inputspace = TypedCoordinateSpace(T, (length(points[1]),))

    return LinOpNFFT(inputspace, outputspace, plan_nufft)
end

function Base.show(io::IO, ::MIME"text/plain", A::LinOpNFFT)
    print(io, "Linear Operator: ")
    println(io, summary(A))
    show(io, A.plan)
    return
end

function apply_!(y, A::LinOpNFFT{I, O, <:PlanNUFFT{T, N, M}}, x) where {T, N, M, I, O}
    return exec_type1!(y, A.plan, x)
end

function apply_adjoint_!(y, A::LinOpNFFT{I, O, <:PlanNUFFT{T, N, M}}, x) where {T, N, M, I, O}
    exec_type2!(y, A.plan, x)  # returns Tuple{output...}; discard and return y directly
    return y
end


end
