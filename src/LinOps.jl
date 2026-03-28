module LinOps

import Adapt
using LinearAlgebra
import LinearAlgebra: mul!
export LinOp, LinOpDiag

VERSION >= v"1.11.0-DEV.469" && eval(
    Meta.parse(
        string(
            "public CoordinateSpace, inputsize, inputspace, isendomorphism, outputsize, outputspace "
        )
    )
)
include("Domains.jl")
include("LinOp.jl")
include("Operations.jl")
include("LinOpDiag.jl")

end
