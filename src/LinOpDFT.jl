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
#=
const _SUPERSCRIPT_DIGITS = Dict(
    '0' => '⁰',
    '1' => '¹',
    '2' => '²',
    '3' => '³',
    '4' => '⁴',
    '5' => '⁵',
    '6' => '⁶',
    '7' => '⁷',
    '8' => '⁸',
    '9' => '⁹',
)

function _superscript_size(sz::NTuple{N, Int}) where {N}
    parts = map(sz) do n
        s = string(n)
        return join(get(_SUPERSCRIPT_DIGITS, c, c) for c in s)
    end
    return join(parts, "ˣ")
end

function Base.summary(A::LinOpDFT)
    return "LinOpDFT ℂ$(_superscript_size(inputsize(A))) ⟶ ℂ$(_superscript_size(outputsize(A)))"
end
 =#
function Base.summary(A::LinOpDFT)
    return "LinOpDFT  $(inputsize(A)) ⟶ $(outputsize(A))"
end


function Base.:*(left::LinOpAdjoint, right::LinOpDFT)
    if parent(left) === right
        return length(inputspace(right)) * I
    end
    return LinOpCompose(left, right)
end

Base.inv(A::LinOpDFT) = eltype(A.forward)(1 / length(inputspace(A))) * adjoint(A)


struct LinOpNFFT{
        I,
        O,
        F,
        D,
    } <: LinOp{I, O}

    inputspace::I
    outputspace::O
    plan::F             # plan for forward transform
    dims::D
    LinOpNFFT(inputspace::I, outputspace::O, plan::F, dims::D) where {I <: AbstractDomain, O <: AbstractDomain, F, D <: NTuple} = new{I, O, F, D}(inputspace, outputspace, plan, dims)
end

function LinOpNFFT(args...; _kwargs...)
    error("Load NonuniformFFTs.jl to use LinOpNFFT")
end
