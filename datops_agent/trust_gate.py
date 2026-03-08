"""
DatOps Agent SDK — Trust Gate middleware

Pre/post tool call authorization and signal reporting.
Mirrors: src/services/dat-agent/src/services/trustGate.ts

External SDK uses local sandbox-level checks (no OPA/authorization service).
This matches the trustGate.ts fallback path (lines 177-183).
"""

import functools
import logging
import time
from typing import Optional, Dict, Any, Callable, TypeVar, Union

from .types import (
    DatOpsConfig,
    GateDecision,
    RiskLevel,
    SandboxLevel,
    SignalEvent,
    ToolBlockedError,
    SANDBOX_ALLOWED_RISKS,
)
from .core import DatOpsCore

logger = logging.getLogger("datops_agent")

F = TypeVar("F", bound=Callable)


class TrustGate:
    """
    Pre/post tool call middleware.

    Before execution:
    - Fetch trust score (cached, 60s TTL)
    - Map to sandbox level (STRICT/ADAPTIVE/OPEN)
    - Check tool risk vs sandbox: STRICT=low, ADAPTIVE=low+medium, OPEN=all
    - Return allow/deny decision

    After execution:
    - Report success/failure signal (fire-and-forget)
    - Signal action field uses tool-specific naming for anti-farming diversity
    """

    def __init__(self, core: DatOpsCore):
        self._core = core

    # ========================================================================
    # Pre-execution gate
    # ========================================================================

    def pre_tool_call(
        self,
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        risk_level: Union[str, RiskLevel] = RiskLevel.MEDIUM,
    ) -> GateDecision:
        """
        Authorize a tool call before execution.

        Returns GateDecision with allowed=True/False and reason.
        Uses local sandbox-level check (no external authorization service).
        """
        # Normalize risk level
        if isinstance(risk_level, str):
            try:
                risk_level = RiskLevel(risk_level.lower())
            except ValueError:
                risk_level = RiskLevel.MEDIUM

        # Fetch trust score (cached)
        trust_result = self._core.get_trust_score()
        trust_score = trust_result.trust_score
        sandbox_level = trust_result.sandbox_level

        # Check if risk level is allowed at current sandbox level
        allowed_risks = SANDBOX_ALLOWED_RISKS.get(sandbox_level, [RiskLevel.LOW])

        if risk_level not in allowed_risks:
            reason = (
                f"Tool '{tool_name}' requires {risk_level.value} risk access, "
                f"but agent is at {sandbox_level.value} sandbox level "
                f"(trust score: {trust_score:.1f}). "
                f"Allowed risk levels: {[r.value for r in allowed_risks]}"
            )
            logger.info(f"Tool blocked: {reason}")
            return GateDecision(
                allowed=False,
                reason=reason,
                sandbox_level=sandbox_level,
                trust_score=trust_score,
            )

        # Check minimum trust threshold from config
        if trust_score < self._core.config.min_trust_for_tool:
            reason = (
                f"Trust score {trust_score:.1f} below minimum "
                f"threshold {self._core.config.min_trust_for_tool}"
            )
            logger.info(f"Tool blocked: {reason}")
            return GateDecision(
                allowed=False,
                reason=reason,
                sandbox_level=sandbox_level,
                trust_score=trust_score,
            )

        # High-risk tool threshold check
        if risk_level == RiskLevel.HIGH:
            if trust_score < self._core.config.trust_threshold_high_risk:
                reason = (
                    f"High-risk tool '{tool_name}' requires trust score >= "
                    f"{self._core.config.trust_threshold_high_risk}, "
                    f"current score: {trust_score:.1f}"
                )
                logger.info(f"Tool blocked: {reason}")
                return GateDecision(
                    allowed=False,
                    reason=reason,
                    sandbox_level=sandbox_level,
                    trust_score=trust_score,
                )

        logger.debug(
            f"Tool '{tool_name}' authorized "
            f"(trust={trust_score:.1f}, sandbox={sandbox_level.value}, "
            f"risk={risk_level.value})"
        )
        return GateDecision(
            allowed=True,
            reason="Authorized by local sandbox-level check",
            sandbox_level=sandbox_level,
            trust_score=trust_score,
        )

    # ========================================================================
    # Post-execution signal
    # ========================================================================

    def post_tool_call(
        self,
        tool_name: str,
        success: bool,
        response_time_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Report a trust signal after tool execution.
        Fire-and-forget — never blocks, never raises.

        Uses tool-specific action naming to avoid anti-farming diversity penalty.
        Without this, all signals have the same action type which triggers
        the 0.25x diversity multiplier after 30 uses.
        """
        event = SignalEvent.ACTION_SUCCESS if success else SignalEvent.ACTION_FAILURE

        # Tool-specific action type for diversity (mirrors trustGate.ts line 203)
        action = f"{tool_name}_{'ok' if success else 'fail'}"

        details: Dict[str, Any] = {
            "source": "datops_agent_sdk",
            "action": action,
            "tool": tool_name,
        }

        if error:
            details["error"] = error[:200]  # Truncate to avoid large payloads

        self._core.report_signal(
            event=event,
            response_time_ms=response_time_ms,
            details=details,
        )

    # ========================================================================
    # Tool wrapper
    # ========================================================================

    def wrap_tool(
        self,
        fn: F,
        tool_name: Optional[str] = None,
        risk_level: Union[str, RiskLevel] = RiskLevel.MEDIUM,
    ) -> F:
        """
        Wrap a callable with trust gate enforcement.

        Before: checks trust score and sandbox level.
        After: reports success/failure signal with response time.

        Usage:
            wrapped = gate.wrap_tool(my_search, "web_search", "medium")
            result = wrapped("query")

        Raises ToolBlockedError if trust check fails.
        """
        name = tool_name or getattr(fn, "__name__", "unknown_tool")

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Pre-execution gate
            decision = self.pre_tool_call(name, kwargs or {}, risk_level)

            if not decision.allowed:
                raise ToolBlockedError(
                    reason=decision.reason,
                    trust_score=decision.trust_score,
                    sandbox_level=decision.sandbox_level,
                )

            # Execute tool with timing
            start = time.monotonic()
            try:
                result = fn(*args, **kwargs)
                elapsed_ms = (time.monotonic() - start) * 1000

                # Post-execution signal (success)
                self.post_tool_call(
                    tool_name=name,
                    success=True,
                    response_time_ms=elapsed_ms,
                )

                return result

            except ToolBlockedError:
                raise  # Don't wrap our own exceptions

            except Exception as e:
                elapsed_ms = (time.monotonic() - start) * 1000

                # Post-execution signal (failure)
                self.post_tool_call(
                    tool_name=name,
                    success=False,
                    response_time_ms=elapsed_ms,
                    error=str(e),
                )

                raise

        return wrapper  # type: ignore[return-value]

    # ========================================================================
    # Batch tool wrapping
    # ========================================================================

    def wrap_tools(
        self,
        tools: Dict[str, Callable],
        risk_levels: Optional[Dict[str, Union[str, RiskLevel]]] = None,
        default_risk: Union[str, RiskLevel] = RiskLevel.MEDIUM,
    ) -> Dict[str, Callable]:
        """
        Wrap multiple tool callables at once.

        Args:
            tools: dict of tool_name → callable
            risk_levels: optional per-tool risk level overrides
            default_risk: default risk level for tools without an override

        Returns:
            dict of tool_name → wrapped callable
        """
        risk_levels = risk_levels or {}
        wrapped = {}
        for name, fn in tools.items():
            risk = risk_levels.get(name, default_risk)
            wrapped[name] = self.wrap_tool(fn, name, risk)
        return wrapped

    # ========================================================================
    # Convenience: current sandbox info
    # ========================================================================

    def get_current_sandbox(self) -> Dict[str, Any]:
        """Get current trust score and sandbox level."""
        trust_result = self._core.get_trust_score()
        allowed_risks = SANDBOX_ALLOWED_RISKS.get(
            trust_result.sandbox_level, [RiskLevel.LOW]
        )
        return {
            "trust_score": trust_result.trust_score,
            "sandbox_level": trust_result.sandbox_level.value,
            "allowed_risk_levels": [r.value for r in allowed_risks],
            "investigation_state": trust_result.investigation_state,
        }
