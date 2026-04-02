abstract type LinOp{I, O} end


inputspace(A::LinOp) = A.inputspace
outputspace(A::LinOp) = A.outputspace

outputspace(A::AbstractMatrix) = CoordinateSpace(size(A, 1))
inputspace(A::AbstractMatrix) = CoordinateSpace(size(A, 2))

inputsize(A::LinOp) = size(inputspace(A))
outputsize(A::LinOp) = size(outputspace(A))
Base.size(A::LinOp) = (outputsize(A), inputsize(A))

Base.eltype(::LinOp) = Bool
outputtype(A::LinOp, x) = typeof(oneunit(eltype(outputspace(A))) * oneunit(eltype(x)))
outputtype(A::UniformScaling, x) = typeof(oneunit(eltype(A)) * oneunit(eltype(x)))
outputtype(A::AbstractMatrix, x) = typeof(oneunit(eltype(A)) * oneunit(eltype(x)))
inputtype(A::LinOp, x) = typeof(oneunit(eltype(inputspace(A))) * oneunit(eltype(x)))

inputtype(::LinOp{I}, _) where {T, I <: TypedCoordinateSpace{T}} = T
outputtype(::LinOp{I, O}, _) where {T, I, O <: TypedCoordinateSpace{T}} = T

inputtype(A::UniformScaling, x) = outputtype(A, x)
inputtype(A::AbstractMatrix, x) = outputtype(A, x)


isendomorphism(A::LinOp) = inputspace(A) === outputspace(A)
isendomorphism(::UniformScaling) = true

function Base.:(==)(a::LinOp, b::LinOp)
    if typeof(a) != typeof(b)
        return false
    end
    return all(getfield(a, f) == getfield(b, f) for f in fieldnames(typeof(b)))
end

function Base.summary(A::LinOp)
    T = typeof(A)
    name = nameof(T)
    return "$(name) $(inputsize(A)) -> $(outputsize(A))"
end

function Base.show(io::IO, A::LinOp)
    return print(io, summary(A))
end

function Base.show(io::IO, ::MIME"text/plain", A::LinOp)
    println(io, "Linear Operator:")
    println(io, summary(A))
    return
end

function assert_applicable(A::LinOp, x)
    return x ∈ inputspace(A) || throw(ArgumentError("The input size (size $(size(x)) ) must belong to the space $(inputspace(A))"))
end

function assert_applicable(A::LinOp, x, y)
    x ∈ inputspace(A) || throw(ArgumentError("The input size (size $(size(x)) ) must belong to the space $(inputspace(A))"))
    return y ∈ outputspace(A) || throw(ArgumentError("The output size (size $(size(y)) ) must belong to the space $(outputspace(A))"))
end

## Applying linear operators
(A::LinOp)(v) = A * v

function mul!(y, A::LinOp, x)
    assert_applicable(A, x, y)
    if applicable(apply_!, y, A, x)
        return apply_!(y, A, x)
    end
    if applicable(apply_, A, x)
        y .= apply_(A, x)
        return y
    end
    throw(ArgumentError("Neither apply_ or apply_! are implemented for $(typeof(A))"))
end

function Base.:*(A::LinOp, x)
    assert_applicable(A, x)
    if applicable(apply_, A, x)
        return apply_(A, x)
    else
        y = similar(x, outputtype(A, x), outputspace(A))
        if applicable(apply_!, y, A, x)
            return apply_!(y, A, x)
        end
    end
    throw(ArgumentError("Neither apply_ or apply_! are implemented for $(typeof(A))"))
end


## Adjoint

struct LinOpAdjoint{I, O, A} <: LinOp{I, O}
    parent::A
    LinOpAdjoint(A::LinOp{O, I}) where {I, O} = new{I, O, typeof(A)}(A)
end

Base.adjoint(A::LinOp) = LinOpAdjoint(A)

Base.parent(A::LinOpAdjoint) = A.parent

LinOpAdjoint(A::LinOpAdjoint) = parent(A)
Base.adjoint(A::LinOpAdjoint) = parent(A)

function Base.summary(A::LinOpAdjoint)
    return "LinOpAdjoint of $(summary(parent(A)))"
end
outputtype(A::LinOpAdjoint, x) = inputtype(parent(A), x)
inputtype(A::LinOpAdjoint, x) = outputtype(parent(A), x)
inputsize(A::LinOpAdjoint) = outputsize(parent(A))
outputsize(A::LinOpAdjoint) = inputsize(parent(A))
inputspace(A::LinOpAdjoint) = outputspace(parent(A))
outputspace(A::LinOpAdjoint) = inputspace(parent(A))


function apply_!(y, A::LinOpAdjoint, x)
    if applicable(apply_adjoint_!, y, parent(A), x)
        return apply_adjoint_!(y, parent(A), x)
    end
    return y .= apply_(A, x)
end


function apply_(A::LinOpAdjoint, x)
    if applicable(apply_adjoint_, parent(A), x)
        return apply_adjoint_(parent(A), x)
    else
        y = similar(x, outputtype(A, x), outputspace(A))
        if applicable(apply_adjoint_!, y, parent(A), x)
            return apply_adjoint_!(y, parent(A), x)
        end
    end
    throw(ArgumentError("Neither apply_adjoint_ or apply_adjoint_! are implemented for $(typeof(A))"))
    # return apply_adjoint_via_ad(parent(A), x) # a voir dans une extension DI
end

function apply_adjoint_!(y, A::LinOpAdjoint, x)
    return apply_!(y, parent(A), x)
end

function apply_adjoint_(A::LinOpAdjoint, x)
    return apply_(parent(A), x)
end

Adapt.adapt_structure(::Any, x::AbstractDomain) = x

#= function Adapt.adapt_structure(to, A::T) where {T <: LinOp}
    return fmap(Adapt.adapt(to), A; exclude = v -> (v isa AbstractDomain) || (v isa AbstractArray))
end =#

function Adapt.adapt_structure(to, A::T) where {T <: LinOp}
    vals = map(fieldnames(T)) do f
        v = getfield(A, f)
        Adapt.adapt(to, v)
    end
    W = Base.typename(T).wrapper
    return W(vals...)
end

function apply_adjoint_via_ad(_, _)
    throw(ArgumentError("Zygote must be loaded to computed adjoint via automatic differentiation"))
end


function ChainRulesCore.rrule(::typeof(Base.:*), A::LinOp, v)
    if applicable(apply_adjoint_, A, v) || applicable(apply_adjoint_!, AbstractArray, A, v)
        ∂Y(Δy) = (NoTangent(), NoTangent(), adjoint(A) * ChainRulesCore.unthunk(Δy))
    else
        return apply_adjoint_via_ad(apply_, A, v)
    end
    return A * v, ∂Y
end
