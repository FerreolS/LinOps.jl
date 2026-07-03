"""
CUDA extension for LinOps FFT operators.

This module provides CUDA-specific `Adapt.adapt_structure` methods so `LinOpDFT`
instances can be moved to GPU arrays and use CUDA FFT plans.
"""
module LinOpsCUDAFFTWExt
using FFTW
using CUDA
import Adapt
import Adapt.adapt_structure
using LinOps #: LinOpDFT, inputsize, outputsize, outputtype, inputtype,AbstractDomain
using LinOps: TypedCoordinateSpace, AbstractDomain, inputtype


"""
    Adapt.adapt_structure(::Type{CUDA.CuArray}, x::LinOpDFT)

Adapt a `LinOpDFT` to a CUDA array backend using the operator input scalar type.
"""
function Adapt.adapt_structure(::Type{CUDA.CuArray}, x::LinOpDFT)
    return Adapt.adapt_structure(CUDA.CuArray{inputtype(x)}, x)
end


"""
    Adapt.adapt_structure(::Type{CUDA.CuArray{T}}, x::LinOpDFT)

Adapt a `LinOpDFT` to CUDA with element type `T`, rebuilding compatible FFT plans.
"""
function Adapt.adapt_structure(::Type{CUDA.CuArray{T}}, x::LinOpDFT) where {T}
    input_sz = inputsize(x)

    if T <: Union{Float32, Float64}
        forward = plan_rfft(
            CUDA.CuArray{T}(undef, input_sz),
            x.dims
        )

        backward = plan_brfft(CUDA.CuArray{Complex{T}}(undef, forward.output_size), input_sz[1], x.dims)
        outputspace = TypedCoordinateSpace(Complex{T}, forward.output_size)
    else
        temp = CUDA.CuArray{T}(undef, input_sz)
        forward = plan_fft(temp, x.dims)
        backward = plan_bfft(temp, x.dims)
        outputspace = TypedCoordinateSpace(T, forward.output_size)
    end


    # Build operator.
    inputspace = TypedCoordinateSpace(T, forward.input_size)
    return LinOpDFT(inputspace, outputspace, x.dims, forward, backward)

end

end
