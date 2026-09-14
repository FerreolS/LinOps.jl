"""
    LinOpSparse{I, O, S}

Linear operator backed by a matrix `S`, mapping domain `I` to domain `O`.

Construct instances with [`LinOpSparse`](@ref), which validates the matrix
dimensions and creates coordinate-space domains.
"""
struct LinOpSparse{I, O, S <: AbstractSparseMatrix} <: LinOp{I, O}
    inputspace::I
    outputspace::O
    sparse_matrix::S
end

"""
    LinOpSparse(matrix, inputsz, outputsz)

Construct a sparse-matrix-backed linear operator with the given input and output
space sizes.
"""
function LinOpSparse(matrix, inputsz, outputsz)
    size(matrix, 1) == prod(outputsz) || throw(DimensionMismatch("Matrix row size does not match output size"))
    size(matrix, 2) == prod(inputsz) || throw(DimensionMismatch("Matrix column size does not match input size"))
    return LinOpSparse(LinOps.CoordinateSpace(inputsz), LinOps.CoordinateSpace(outputsz), matrix)
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
