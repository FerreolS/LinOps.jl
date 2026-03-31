module LinOps

using AbstractFFTs,
    Adapt,
    ArrayTools,
    KernelAbstractions,
    LinearAlgebra,
    StaticArrays

import LinearAlgebra:
    mul!,
    I,
    UniformScaling

export I,
    LinOp,
    LinOpDFT,
    LinOpDiag,
    LinOpMapslice,
    LinOpNFFT,
    UniformScaling,
    inputsize,
    mul!,
    outputsize

VERSION >= v"1.11.0-DEV.469" && eval(
    Meta.parse(
        string(
            "public CoordinateSpace,  isendomorphism, outputspace, inputspace "
        )
    )
)
include("Domains.jl")
include("LinOp.jl")
include("Operations.jl")
include("LinOpDiag.jl")
include("LinOpDFT.jl")
include("LinOpMapslice.jl")

end
