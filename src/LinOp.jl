abstract type LinOp{I, O} end
inputspace(A::LinOp) = A.inputspace
outputspace(A::LinOp) = A.outputspace

inputsize(A::LinOp) = size(inputspace(A))
outputsize(A::LinOp) = size(outputspace(A))
isendomorphism(A::LinOp) = inputspace(A) === outputspace(A)

(A::LinOp)(v) = A * v


Base.:+(A::LinOp, B::LinOp) = add(A, B)
Base.:+(A::LinOp, B) = add(A, B)
Base.:+(B, A::LinOp) = add(A, B)
Base.:-(A::LinOp, B::LinOp) = add(A, -B)
Base.:-(A::LinOp, B) = add(A, -B)
Base.:-(A, B::LinOp) = add(A, -B)
Base.:-(A::LinOp) = -1 * A

Base.:*(A::LinOp, v) = apply(A, v)

Base.:*(A::LinOp, B::LinOp) = compose(A, B)
Base.:∘(A::LinOp, B::LinOp) = compose(A, B)
Base.:^(A::LinOp, n::Int) = n > 0 ? compose(A, A^(n - 1)) : (n == 0 ? LinearAlgebra.I : inverse(A)^(-n))

Base.inv(A::LinOp) = inverse(A)

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

function Base.:/(A::LinOp, B::Number)
    return A * inv(B)
end
function Base.:\(A::Number, B::LinOp)
    return inv(A) * B
end
