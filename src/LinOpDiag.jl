"""
    LinOpDiag(diag)

Diagonal linear operator with coefficients from `diag`.

Applies elementwise multiplication in forward mode and conjugate multiplication in
adjoint mode.

# Examples
```julia
A = LinOpDiag([1.0, 2.0, 3.0])
y = A * [10.0, 10.0, 10.0]   # [10.0, 20.0, 30.0]
```
"""
struct LinOpDiag{I, D <: AbstractArray} <: LinOp{I, I}
    inputspace::I
    diag::D
end

function LinOpDiag(diag::D) where {D <: AbstractArray}
    inspace = CoordinateSpace(size(diag))
    return LinOpDiag(inspace, diag)
end

outputspace(A::LinOpDiag) = inputspace(A)
isendomorphism(::LinOpDiag) = true

function Base.summary(A::LinOpDiag)
    return "LinOpDiag ($(eltype(A.diag))) $(inputsize(A)) -> $(outputsize(A))"
end

apply_(A::LinOpDiag, x) = A.diag .* x
apply_adjoint_(A::LinOpDiag, x) = conj.(A.diag) .* x

apply_!(y, A::LinOpDiag, x) = @. y = A.diag * x
apply_adjoint_!(y, A::LinOpDiag, x) = @. y = conj.(A.diag) * x

# This make A'' == A and not === .Not sure if this is better than the default where A'' === A
#= function LinOpAdjoint(A::LinOpDiag)
    return LinOpDiag(conj.(A.diag))
end
 =#

function LinOpCompose(A::D, B::LinOpAdjoint{O, I, D}) where {I, O, D <: LinOpDiag{I, O}}
    if parent(B) === A
        return LinOpDiag(abs2.(A.diag))
    end
    return LinOpDiag(@. A.diag * conj(parent(B).diag))
end

function LinOpCompose(A::LinOpAdjoint{I, O, D}, B::D) where {I, O, D <: LinOpDiag{O, I}}
    if parent(A) === B
        return LinOpDiag(abs2.(B.diag))
    end
    return LinOpDiag(@. B.diag * conj(parent(A).diag))
end

function LinOpCompose(A::LinOpDiag, B::LinOpDiag)
    outputspace(B) == inputspace(A) || throw(ArgumentError("The output space of the right operator should match the input space of the left operator"))
    if A.diag == inv.(B.diag)
        return I
    end
    return LinOpDiag(@. A.diag * B.diag)
end

function LinOpCompose(A::Number, B::LinOpDiag)
    if A == 0
        return 0I
    end
    if A == 1
        return B
    end
    return LinOpDiag(@. A * B.diag)
end

function LinOpCompose(A::UniformScaling, B::LinOpDiag)
    if A == UniformScaling(0)
        return A
    end
    if A == UniformScaling(1)
        return B
    end
    return LinOpDiag(A * B.diag)
end


Base.inv(A::LinOpDiag) = LinOpDiag(@. 1 / A.diag)
Base.:^(A::LinOpDiag, n::Int) = LinOpDiag(@. A.diag^n)

function LinOpSum(A::LinOpDiag, B::LinOpDiag)
    inputspace(B) == inputspace(A) || throw(ArgumentError("The input space of operators should match"))
    if A.diag == -B.diag
        return 0I
    end
    return LinOpDiag(@. A.diag + B.diag)
end

LinOpSum(A::Number, B::LinOpDiag) = LinOpDiag(@. A + B.diag)
LinOpSum(A::UniformScaling, B::LinOpDiag) = LinOpDiag(@. A.λ + B.diag)
