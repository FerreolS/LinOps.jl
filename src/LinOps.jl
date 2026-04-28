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
