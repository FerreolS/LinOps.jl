module LinOps

import LinearAlgebra: mul!, UniformScaling

export LinOp, LinOpDiag

public CoordinateSpace, inputspace, outputspace, inputsize, outputsize, isendomorphism

include("Domains.jl")
include("LinOp.jl")
include("Operations.jl")
include("LinOpDiag.jl")

end
