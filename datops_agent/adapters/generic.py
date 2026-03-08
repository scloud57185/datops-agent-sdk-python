"""
DatOps Agent SDK — Generic decorator adapter

Framework-agnostic decorator for trust-gated tool execution.

Usage:
    from datops_agent import DatOps

    datops = DatOps(api_key="dat_xxx")

    @datops.trust_gate(risk_level="medium")
    def search_web(query: str) -> str:
        return requests.get(f"https://api.search.com?q={query}").text

    # Tool call is now trust-gated
    result = search_web("weather in NYC")
"""

import functools
import logging
import time
from typing import Any, Callable, Optional, TypeVar, Union

from ..types import RiskLevel, ToolBlockedError

logger = logging.getLogger("datops_agent")

F = TypeVar("F", bound=Callable)


def trust_gate_decorator(
    trust_gate,
    risk_level: Union[str, RiskLevel] = RiskLevel.MEDIUM,
    tool_name: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator factory for trust-gated function execution.

    Args:
        trust_gate: TrustGate instance
        risk_level: Risk level for this tool (low/medium/high)
        tool_name: Optional override for tool name (defaults to function name)

    Returns:
        Decorator that wraps function with trust enforcement
    """
    if isinstance(risk_level, str):
        try:
            risk_level = RiskLevel(risk_level.lower())
        except ValueError:
            risk_level = RiskLevel.MEDIUM

    def decorator(fn: F) -> F:
        name = tool_name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Pre-execution gate
            decision = trust_gate.pre_tool_call(name, kwargs or {}, risk_level)

            if not decision.allowed:
                raise ToolBlockedError(
                    reason=decision.reason,
                    trust_score=decision.trust_score,
                    sandbox_level=decision.sandbox_level,
                )

            # Execute with timing
            start = time.monotonic()
            try:
                result = fn(*args, **kwargs)
                elapsed_ms = (time.monotonic() - start) * 1000

                trust_gate.post_tool_call(
                    tool_name=name,
                    success=True,
                    response_time_ms=elapsed_ms,
                )

                return result

            except ToolBlockedError:
                raise

            except Exception as e:
                elapsed_ms = (time.monotonic() - start) * 1000

                trust_gate.post_tool_call(
                    tool_name=name,
                    success=False,
                    response_time_ms=elapsed_ms,
                    error=str(e),
                )

                raise

        # Attach metadata for introspection
        wrapper._datops_wrapped = True  # type: ignore[attr-defined]
        wrapper._datops_risk_level = risk_level  # type: ignore[attr-defined]
        wrapper._datops_tool_name = name  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator
