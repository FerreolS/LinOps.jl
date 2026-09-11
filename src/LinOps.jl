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

import Adapt
using Adapt: adapt
using ArrayTools: colons
import ChainRulesCore
using ChainRulesCore: NoTangent, unthunk
import KernelAbstractions
using KernelAbstractions: @index, @kernel, get_backend, synchronize
using LinearAlgebra: I, UniformScaling, diag, dot
import LinearAlgebra: mul!
using StaticArrays: SVector
using TypeUtils: adapt_precision, parameterless

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
    LinOpSparse,
    LinOpMapslice,
    LinOpNFFT,
    UniformScaling,
    inputsize,
    mul!,
    outputsize

VERSION >= v"1.11.0-DEV.469" && eval(
    Meta.parse(
        string(
            "public AbstractDomain, CoordinateSpace, LinOpAdjoint, apply_, apply_!, apply_adjoint_, apply_adjoint_!, outputtype, inputspace, outputspace, isendomorphism, promote_domain, ⊂ "
        )
    )
)
include("Domains.jl")
include("LinOp.jl")
include("utils.jl")
include("Operations.jl")
include("LinOpDiag.jl")
include("LinOpGrad.jl")
include("LinOpDFT.jl")
include("LinOpMapslice.jl")
include("LinOpSparse.jl")

end
