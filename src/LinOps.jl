"""
    LinOps

Linear-operator toolkit with explicit input/output domains.

`LinOps` provides composable linear operators (`LinOp` subtypes), algebra on operators
(`*`, `+`, adjoint, inverse when available), and optional backend-dependent operators
such as `LinOpDFT` and `LinOpNFFT`.

The package also exposes public extension points so users can define custom operators by
subtyping `LinOp` and implementing `apply_` / `apply_!` and adjoint variants.
"""
module LinOps

using AbstractFFTs,
    Adapt,
    ArrayTools,
    ChainRulesCore,
    KernelAbstractions,
    LinearAlgebra,
    StaticArrays

import LinearAlgebra:
    mul!,
    I,
    UniformScaling

@doc "Identity operator from `LinearAlgebra` used in LinOps operator algebra." I
@doc "Uniform-scaling operator type from `LinearAlgebra` used in LinOps compositions and sums." UniformScaling
@doc "In-place linear application from `LinearAlgebra`; LinOps extends it for `LinOp` objects." mul!

export I,
    has_operator,
    operator_backend,
    LinOp,
    LinOpDFT,
    LinOpDiag,
    LinOpGrad,
    LinOpMapslice,
    LinOpNFFT,
    UniformScaling,
    inputsize,
    mul!,
    outputsize

VERSION >= v"1.11.0-DEV.469" && eval(
    Meta.parse(
        string(
            "public AbstractDomain, CoordinateSpace, TypedCoordinateSpace, LinOpAdjoint, apply_, apply_!, apply_adjoint_, apply_adjoint_!, inputtype, outputtype, inputspace, outputspace, isendomorphism, promote_domain, ⊂ "
        )
    )
)
include("Domains.jl")
include("LinOp.jl")
include("Operations.jl")
include("LinOpDiag.jl")
include("LinOpGrad.jl")
include("LinOpDFT.jl")
include("LinOpMapslice.jl")

end
