module LinOpsFFTWExt
import FFTW
import FFTW: fftwComplex, fftwReal, plan_brfft, plan_rfft, plan_fft, plan_bfft
import LinOps
import LinOps:
    CoordinateSpace, LinOpDFT, apply_, apply_!, apply_adjoint_, apply_adjoint_!, outputtype,
    inputsize, outputsize


const PLANNING = (
    FFTW.ESTIMATE | FFTW.MEASURE | FFTW.PATIENT |
        FFTW.EXHAUSTIVE | FFTW.WISDOM_ONLY
)


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

    inputspace = CoordinateSpace(forward.sz)
    outputspace = CoordinateSpace(forward.osz)
    return LinOpDFT(inputspace, outputspace, forward, backward)
end


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

    inputspace = CoordinateSpace(forward.sz)
    outputspace = CoordinateSpace(forward.osz)
    return LinOpDFT(inputspace, outputspace, forward, backward)
end

apply_!(y, A::LinOpDFT, x) = FFTW.mul!(y, A.forward, complex(x))
apply_adjoint_!(y, A::LinOpDFT, x) = FFTW.mul!(y, A.backward, complex(x))

outputtype(A::LinOpDFT{I, O, <:FFTW.FFTWPlan{T}}, x) where {I, O, T} = typeof(oneunit(T) * oneunit(eltype(x)))

function Base.summary(A::LinOpDFT{I, O, <:FFTW.FFTWPlan{T}}) where {I, O, T}
    return "LinOpDFT ($T) $(inputsize(A)) -> $(outputsize(A))"
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
    flags == planning ||
        bad_argument("only FFTW planning flags can be specified")
    return UInt32(planning)
end


end
