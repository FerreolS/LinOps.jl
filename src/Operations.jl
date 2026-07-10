## Composition
Base.:*(A::LinOp, B::LinOp) = LinOpCompose(A, B)
Base.:*(A::LinOp, B::Union{<:UniformScaling, <:Number}) = B * A
Base.:*(A::Number, B::LinOp) = UniformScaling(A) * B
Base.:*(A::UniformScaling, B::LinOp) = LinOpCompose(A, B)
Base.:∘(A::LinOp, B) = LinOpCompose(A, B)
Base.:∘(A, B::LinOp) = LinOpCompose(A, B)
Base.:∘(A::LinOp, B::LinOp) = LinOpCompose(A, B)


struct LinOpCompose{I, O, L <: Union{UniformScaling, LinOp}, R <: LinOp} <: LinOp{I, O}
    inputspace::I
    outputspace::O
    left::L
    right::R
end

Base.eltype(A::LinOpCompose) = eltype(oneunit(eltype(A.left)) * oneunit(eltype(A.right)))

function LinOpCompose(A::LinOp, B::LinOp)
    outputspace(B) ⊂ inputspace(A) || throw(ArgumentError("The output space of the right operator should match the input space of the left operator"))

    TIB = get_type(inputspace(B))
    TIA = get_type(inputspace(A))
    TOA = get_type(outputspace(A))
    TOB = get_type(outputspace(B))

    SIA = get_storage(inputspace(A))
    SIB = get_storage(inputspace(B))
    SOA = get_storage(outputspace(A))
    SOB = get_storage(outputspace(B))

    NIA = ndims(inputspace(A))
    NOA = ndims(outputspace(A))

    Ti = !isconcretetype(TIB) ? TIA : TIB
    Si = !isconcretetype(SIB{Ti, NIA}) ? SIA : SIB
    inspace = CoordinateSpace(Ti, inputsize(B), Si)

    To = !isconcretetype(TOA) ? TOB : TOA
    So = !isconcretetype(SOA{To, NOA}) ? SOB : SOA
    outspace = CoordinateSpace(To, outputsize(A), So)

    return LinOpCompose(inspace, outspace, A, B)
end

LinOpCompose(A::LinOp, B::UniformScaling) = B * A

function LinOpCompose(A::UniformScaling, B::LinOp)
    if A == UniformScaling(0)
        return 0
    end
    if A == UniformScaling(1)
        return B
    end
    return LinOpCompose(inputspace(B), outputspace(B), A, B)
end


function LinOpCompose(A::LinOpCompose, B::LinOp)
    return A.left * (A.right * B)
end

function LinOpCompose(A::LinOpCompose, B::LinOpCompose{I, O, <:UniformScaling}) where {I, O}
    return A.left * (A.right * B)
end

function LinOpCompose(A::LinOp, B::LinOpCompose{I, O, <:UniformScaling}) where {I, O}
    return B.left * A * B.right
end

function LinOpCompose(A::UniformScaling, B::LinOpCompose{I, O, <:UniformScaling}) where {I, O}
    C = A * B.left
    if C == UniformScaling(0)
        return 0
    end
    if C == UniformScaling(1)
        return B.right
    end
    return C * B.right
end

LinOpAdjoint(A::LinOpCompose) = A.right' * A.left'

function apply_(A::LinOpCompose, x)
    return A.left * (A.right * x)
end

function apply_(A::LinOpCompose{I, O, <:UniformScaling}, x) where {I, O}
    return A.left.λ * (A.right * x)
end

function apply_!(y, A::LinOpCompose{I, O, <:UniformScaling}, x) where {I, O}
    return apply_!(y, A.right, A.left * x)
end

function apply_!(y, A::LinOpCompose, x)
    return apply_!(y, A.left, A.right * x)
end

function apply_adjoint_(A::LinOpCompose, x)
    return A.right' * (A.left' * x)
end

function apply_adjoint_!(y, A::LinOpCompose, x)
    return apply_adjoint_!(y, A.right, A.left' * x)
end

Base.inv(A::LinOpCompose) = Base.inv(A.right) * Base.inv(A.left)

## Inverse
Base.:^(A::LinOp, n::Int) = n > 0 ? A^(n - 1) * A : (n == 0 ? LinearAlgebra.I : Base.inv(A)^(-n))


function Base.:/(A::Union{LinOp, Number, UniformScaling}, B::LinOp)
    return A * inv(B)
end

function Base.:/(A::LinOp, B::LinOp)
    if A === B
        return LinearAlgebra.I
    end
    return A * inv(B)
end

function Base.:/(A::LinOp, B::Union{LinOp, Number, UniformScaling})
    return A * inv(B)
end

function Base.:\(A::Union{LinOp, Number, UniformScaling}, B::LinOp)
    return inv(A) * B
end

function Base.:\(A::LinOp, B::Union{LinOp, Number, UniformScaling})
    return inv(A) * B
end

function Base.:\(A::LinOp, B::LinOp)
    if A === B
        return LinearAlgebra.I
    end
    return inv(A) * B
end

## Sum
Base.:+(A::Union{LinOp, Number, UniformScaling}, B::LinOp) = LinOpSum(A, B)
Base.:+(A::LinOp, B::Union{UniformScaling, Number}) = B + A

struct LinOpSum{I, O, L <: Union{UniformScaling, LinOp}, R <: LinOp} <: LinOp{I, O}
    inputspace::I
    outputspace::O
    left::L
    right::R
end

Base.eltype(A::LinOpSum) = eltype(oneunit(eltype(A.left)) + oneunit(eltype(A.right)))

function LinOpSum(A::LinOp, B::LinOp)
    inputspace(A) == inputspace(B) || throw(ArgumentError("The input spaces of the two operators should match"))
    outputspace(A) == outputspace(B) || throw(ArgumentError("The output spaces of the two operators should match"))
    return LinOpSum(inputspace(A), outputspace(A), A, B)
end

function LinOpSum(A::Number, B::LinOp)
    if A == 0
        return B
    end
    return UniformScaling(A) + B
end

function LinOpSum(A::UniformScaling, B::LinOp)
    if A == UniformScaling(0)
        return B
    end
    return LinOpSum(inputspace(B), outputspace(B), A, B)
end


function apply_(A::LinOpSum, x)
    return A.left * x + A.right * x
end

function apply_!(y, A::LinOpSum, x)
    apply_!(y, A.right, x)
    y .+= A.left * x
    return y
end

function apply_adjoint_(A::LinOpSum, x)
    return A.left' * x + A.right' * x
end

function apply_adjoint_!(y, A::LinOpSum, x)
    apply_adjoint_!(y, A.right, x)
    y .+= A.left' * x
    return y
end

function Base.:-(A::LinOp, B::LinOp)
    if A === B
        return 0
    end
    return A + (-1 * B)
end
Base.:-(A::LinOp, B::Union{LinOp, Number, UniformScaling}) = A + (-1 * B)
Base.:-(A::Union{LinOp, Number, UniformScaling}, B::LinOp) = A + (-1 * B)
Base.:-(A::LinOp) = -1 * A
