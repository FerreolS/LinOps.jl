import AbstractFFTs: Plan

struct LinOpDFT{
        I, O,
        F <: Plan,     # type of forward plan
        B <: Plan,      # type of backward plan
    } <: LinOp{I, O}
    inputspace::I
    outputspace::O
    forward::F             # plan for forward transform
    backward::B            # plan for backward transform
    LinOpDFT(inputspace::I, outputspace::O, forward::F, backward::B) where {I, O, F, B} = new{I, O, F, B}(inputspace, outputspace, forward, backward)
    # unitary::Bool ?
end

function LinOpDFT(args...; _kwargs...)
    error("Load FFTW to use LinOpDFT")
end


apply_(A::LinOpDFT, v) = A.forward * v
apply_adjoint_(A::LinOpDFT, v) = A.backward * v


function Base.:*(left::LinOpDFT, right::LinOpAdjoint)
    if parent(right) === left
        return length(inputspace(right)) * I
    end
    return LinOpCompose(left, right)
end

Base.eltype(::LinOpDFT) = ComplexF64

function Base.:*(left::LinOpAdjoint, right::LinOpDFT)
    if parent(left) === right
        return length(inputspace(right)) * I
    end
    return LinOpCompose(left, right)
end


struct LinOpNFFT{
        I,
        O,
        F,
        N,
    } <: LinOp{I, O}

    inputspace::I
    outputspace::O
    plan::F             # plan for forward transform
    dims::NTuple{N, Int}  # dimensions along which the transform is applied

    LinOpNFFT(inputspace::I, outputspace::O, plan::F, dims::NTuple{N, Int}) where {I <: CoordinateSpace, O <: CoordinateSpace, F, N} = new{I, O, F, N}(inputspace, outputspace, plan, dims)
end


function LinOpNFFT(args...; _kwargs...)
    error("Load NonuniformFFTs.jl to use LinOpNFFT")
end
