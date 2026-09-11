struct LinOpSparse{I, O, S <: AbstractMatrix} <: LinOp{I, O}
    inputspace::I
    outputspace::O
    sparse_matrix::S
end

function LinOpSparse(matrix, sizein, sizeout)
    size(matrix, 1) == prod(sizeout) || throw(DimensionMismatch("Matrix row size does not match output size"))
    size(matrix, 2) == prod(sizein) || throw(DimensionMismatch("Matrix column size does not match input size"))
    return LinOpSparse(LinOps.CoordinateSpace(sizein), LinOps.CoordinateSpace(sizeout), matrix)
end

Base.eltype(A::LinOpSparse) = eltype(A.sparse_matrix)


function LinOps.apply_(A::LinOpSparse, x)
    return reshape(A.sparse_matrix * reshape(x, :), size(A.outputspace))
end

function LinOps.apply_!(y, A::LinOpSparse, x)
    ry = reshape(y, :)
    mul!(ry, A.sparse_matrix, reshape(x, :))
    return y
end

function LinOps.apply_adjoint_(A::LinOpSparse, x)
    return reshape(A.sparse_matrix' * reshape(x, :), size(A.inputspace))
end

function LinOps.apply_adjoint_!(y, A::LinOpSparse, x)
    ry = reshape(y, :)
    mul!(ry, A.sparse_matrix', reshape(x, :))
    return y
end
