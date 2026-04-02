module LinOpsZygoteExt

import LinOps: LinOp, apply_adjoint_via_ad, apply_
import Zygote: ZygoteRuleConfig
import ChainRulesCore: rrule_via_ad

apply_adjoint_via_ad(A::LinOp, v) = rrule_via_ad(ZygoteRuleConfig(), apply_, A, v)[1]

end
