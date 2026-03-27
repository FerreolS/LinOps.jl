struct LinOpDiag{I, D <: AbstractArray} <: LinOp{I, I}
    inputspace::I
    diag::D
    function LinOpDiag(diag::D) where {D <: AbstractArray}
        inspace = CoordinateSpace(size(diag))
        return new{typeof(inspace), D}(inspace, diag)
    end
end

outputspace(A::LinOpDiag) = inputspace(A)
Base.eltype(A::LinOpDiag) = eltype(A.diag)
isendomorphism(::LinOpDiag) = true

apply_(A::LinOpDiag, x) = return A.diag .* x
apply_adjoint_(A::LinOpDiag, x) = return conj.(A.diag) .* x

apply_!(y, A::LinOpDiag, x) = @. y = A.diag * x
apply_adjoint_!(y, A::LinOpDiag, x) = @. y = conj.(A.diag) * x

LinOpCompose(A::LinOpDiag, B::LinOpDiag) = LinOpDiag(@. A.diag * B.diag)
LinOpCompose(A::Union{Number, UniformScaling}, B::LinOpDiag) = LinOpDiag(@. A * B.diag)


Base.inv(A::LinOpDiag) = LinOpDiag(@. 1 / A.diag)
Base.:^(A::LinOpDiag, n::Int) = LinOpDiag(@. A.diag^n)

LinOpSum(A::LinOpDiag, B::LinOpDiag) = LinOpDiag(@. A.diag + B.diag)
LinOpSum(A::Union{Number, UniformScaling}, B::LinOpDiag) = LinOpDiag(@. A + B.diag)
