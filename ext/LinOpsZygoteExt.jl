"""
Zygote extension for LinOps adjoint fallback.

This module defines AD-based adjoint computation when explicit adjoint methods are
not implemented for an operator.
"""
module LinOpsZygoteExt

import LinOps: LinOp, apply_adjoint_via_ad, apply_
import Zygote: ZygoteRuleConfig
import ChainRulesCore: rrule_via_ad

"""Compute adjoint action via Zygote/ChainRules automatic differentiation."""
apply_adjoint_via_ad(A::LinOp, v) = rrule_via_ad(ZygoteRuleConfig(), apply_, A, v)[1]

end
