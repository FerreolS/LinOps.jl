"""
FFTW extension for LinOps optional DFT operators.

This module activates `has_operator(:dft)` and provides FFTW-backed `LinOpDFT`
constructors and array adaptation methods.
"""
module LinOpsFFTWExt
import Adapt
import Adapt: adapt_structure
import FFTW
import FFTW: fftwComplex, fftwReal, fftwNumber, plan_brfft, plan_rfft, plan_fft, plan_bfft
import LinOps
import LinOps: TypedCoordinateSpace, LinOpDFT, apply_!, apply_adjoint_!, inputsize, outputsize,
    outputtype, LinOpAdjoint

LinOps.has_operator(::Val{:dft}) = true
LinOps.operator_backend(::Val{:dft}) = :fftw


const PLANNING = (
    FFTW.ESTIMATE | FFTW.MEASURE | FFTW.PATIENT |
        FFTW.EXHAUSTIVE | FFTW.WISDOM_ONLY
)


"""
    LinOpDFT(::Type{T}, sz; dims=1:N, timelimit=FFTW.NO_TIMELIMIT, flags=FFTW.MEASURE)

Create an FFTW-backed real-to-complex DFT operator for array shape `sz`.
"""
# Real-to-complex FFT.
function LinOpDFT(
        ::Type{T},
        sz::NTuple{N, Int};
        dims = 1:N,
        timelimit::Real = FFTW.NO_TIMELIMIT,
        flags::Integer = FFTW.MEASURE
    ) where {T <: fftwReal, N}
    # Check arguments and build dimension list of the result of the forward
    # real-to-complex (r2c) transform.
    planning = check_flags(flags)


    # Compute the plans with suitable FFTW flags.  The forward transform (r2c)
    # must preserve its input, while the backward transform (c2r) may destroy
    # it (in fact there are no input-preserving algorithms for
    # multi-dimensional c2r transforms implemented in FFTW, see
    # http://www.fftw.org/doc/Planner-Flags.html).
    forward = plan_rfft(
        Array{T}(undef, sz), dims;
        flags = (planning | FFTW.PRESERVE_INPUT),
        timelimit = timelimit
    )
    backward = plan_brfft(
        Array{Complex{T}}(undef, forward.osz), sz[1], dims;
        flags = (planning | FFTW.DESTROY_INPUT),
        timelimit = timelimit
    )

    # Build operator.

    inputspace = TypedCoordinateSpace(T, forward.sz)
    outputspace = TypedCoordinateSpace(Complex{T}, forward.osz)
    return LinOpDFT(inputspace, outputspace, forward, backward)
end


"""
    LinOpDFT(::Type{T}, sz; dims=1:N, timelimit=FFTW.NO_TIMELIMIT, flags=FFTW.MEASURE)

Create an FFTW-backed complex-to-complex DFT operator for array shape `sz`.
"""
# Complex-to-complex FFT.
function LinOpDFT(
        ::Type{T},
        sz::NTuple{N, Int};
        dims = 1:N,
        timelimit::Real = FFTW.NO_TIMELIMIT,
        flags::Integer = FFTW.MEASURE
    ) where {T <: fftwComplex, N}
    # Check arguments.  The input and output of the complex-to-complex
    # transform have the same dimensions.
    planning = check_flags(flags)
    temp = Array{T}(undef, sz)

    # Compute the plans with suitable FFTW flags.  For maximum efficiency, the
    # transforms are always applied in-place and thus cannot preserve their
    # inputs.
    forward = plan_fft(
        temp,
        dims;
        flags = (planning | FFTW.DESTROY_INPUT),
        timelimit = timelimit
    )
    backward = plan_bfft(
        temp,
        dims;
        flags = (planning | FFTW.DESTROY_INPUT),
        timelimit = timelimit
    )

    # Build operator.

    inputspace = TypedCoordinateSpace(T, forward.sz)
    outputspace = TypedCoordinateSpace(T, forward.osz)
    return LinOpDFT(inputspace, outputspace, forward, backward)
end

LinOpDFT(sz::NTuple; kwargs...) = LinOpDFT(ComplexF64, sz; kwargs...)

apply_!(y, A::LinOpDFT, x) = FFTW.mul!(y, A.forward, complex(x))
apply_adjoint_!(y, A::LinOpDFT, x) = FFTW.mul!(y, A.backward, complex(x))

function Base.summary(A::LinOpDFT{I, O, <:FFTW.FFTWPlan{T}}) where {I, O, T}
    return "LinOpDFT ($T) $(inputsize(A)) -> $(outputsize(A))"
end
#=

function Adapt.adapt_structure(::Type{A}, x::LinOpDFT) where {A <: AbstractArray}
    return Adapt.adapt_structure(A{eltype(inputspace(x))}, x)
end
 =#

function Adapt.adapt_structure(::Type{A}, x::LinOpDFT) where {T <: fftwNumber, A <: AbstractArray{T}}
    sz = inputsize(x)
    planning = planning = check_flags(FFTW.MEASURE)
    timelimit = FFTW.NO_TIMELIMIT
    # Compute the plans with suitable FFTW flags.  For maximum efficiency, the
    # transforms are always applied in-place and thus cannot preserve their
    # inputs.

    if T <: fftwReal
        forward = plan_rfft(
            Array{T}(undef, sz);
            flags = (planning | FFTW.PRESERVE_INPUT),
            timelimit = timelimit
        )

        backward = plan_brfft(
            Array{Complex{T}}(undef, forward.osz), sz[1];
            flags = (planning | FFTW.DESTROY_INPUT),
            timelimit = timelimit
        )
    else
        temp = Array{T}(undef, sz)
        forward = plan_fft(
            temp; flags = (planning | FFTW.DESTROY_INPUT),
            timelimit = timelimit
        )
        backward = plan_bfft(
            temp; flags = (planning | FFTW.DESTROY_INPUT),
            timelimit = timelimit
        )
    end


    # Build operator.
    inputspace = TypedCoordinateSpace(T, forward.sz)
    outputspace = TypedCoordinateSpace(T, forward.osz)
    return LinOpDFT(inputspace, outputspace, forward, backward)

end


#------------------------------------------------------------------------------
# Utilities borrowed from LazyAlgebra

"""

`check_flags(flags)` checks whether `flags` is an allowed bitwise-or
combination of FFTW planner flags (see
http://www.fftw.org/doc/Planner-Flags.html) and returns the filtered flags.

"""
function check_flags(flags::Integer)
    planning = flags & PLANNING
    flags == planning || throw(ArgumentError("only FFTW planning flags can be specified"))
    return UInt32(planning)
end


end
