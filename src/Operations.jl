## Composition
Base.:*(A::LinOp, B::LinOp) = LinOpCompose(A, B)
Base.:*(A::LinOp, B::Union{UniformScaling, Number}) = LinOpCompose(B, A)
Base.:*(A::Union{UniformScaling, Number}, B::LinOp) = LinOpCompose(A, B)
Base.:∘(A::LinOp, B) = LinOpCompose(A, B)
Base.:∘(A, B::LinOp) = LinOpCompose(A, B)
Base.:∘(A::LinOp, B::LinOp) = LinOpCompose(A, B)


struct LinOpCompose{I, O, L <: Union{UniformScaling, LinOp}, R <: LinOp} <: LinOp{I, O}
    inputspace::I
    outputspace::O
    left::L
    right::R
end


function LinOpCompose(A::LinOp, B::LinOp)
    outputspace(B) == inputspace(A) || throw(ArgumentError("The output space of the right operator should match the input space of the left operator"))
    return LinOpCompose(inputspace(B), outputspace(A), A, B)
end


function LinOpCompose(A::Number, B::LinOp)
    if A == 0
        return 0
    end
    if A == 1
        return B
    end
    return LinOpCompose(inputspace(B), outputspace(B), UniformScaling(A), B)
end

function LinOpCompose(A::UniformScaling, B::LinOp)
    if A == UniformScaling(0)
        return false
    end
    if A == UniformScaling(1)
        return B
    end
    return LinOpCompose(inputspace(B), outputspace(B), A, B)
end


function LinOpCompose(A::LinOpCompose, B::LinOp)
    return A.left * (A.right * B)
end

function LinOpCompose(A::LinOp, B::LinOpCompose{I, O, UniformScaling}) where {I, O}
    return B.left * (A * B.right)
end

function LinOpCompose(A::UniformScaling, B::LinOpCompose{I, O, UniformScaling}) where {I, O}
    C = A * B.left
    if C == UniformScaling(0)
        return false
    end
    if C == UniformScaling(1)
        return B.right
    end
    return C * B.right
end


function apply_(A::LinOpCompose{I, O, UniformScaling}, x) where {I, O}
    return A.left * apply_(A.right, x)
end

function apply_(A::LinOpCompose, x)
    return apply_(A.left, apply_(A.right, x))
end

function apply_!(y, A::LinOpCompose{I, O, UniformScaling}, x) where {I, O}
    return apply_!(y, A.right, A.left * x)
end

function apply_!(y, A::LinOpCompose, x)
    return apply_!(y, A.left, apply_(A.right, x))
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


function Base.:/(A::T, B::LinOp) where {T}
    if A === B
        return LinearAlgebra.I
    end
    return A * inv(B)
end
function Base.:\(B::LinOp, A::T) where {T}
    if A === B
        return LinearAlgebra.I
    end
    return inv(B) * A
end

function Base.:/(A::LinOp, B)
    return A * inv(B)
end
function Base.:\(A, B::LinOp)
    return inv(A) * B
end


## Sum
Base.:+(A, B::LinOp) = LinOpSum(A, B)
Base.:+(A::LinOp, B::Union{UniformScaling, Number}) = B + A

struct LinOpSum{I, O, L <: Union{UniformScaling, LinOp}, R <: LinOp} <: LinOp{I, O}
    inputspace::I
    outputspace::O
    left::L
    right::R
end

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

Base.:-(A::LinOp, B::LinOp) = A + (-1 * B)
Base.:-(A::LinOp, B) = A + (-1 * B)
Base.:-(A, B::LinOp) = A + (-1 * B)
Base.:-(A::LinOp) = -1 * A
