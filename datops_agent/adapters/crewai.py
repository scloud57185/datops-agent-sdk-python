"""
DatOps Agent SDK — CrewAI adapter

Wraps CrewAI tools with trust-gated execution.

Usage:
    from datops_agent import DatOps
    crew = DatOps.wrap_crewai(my_crew, api_key="dat_xxx")
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

from ..types import RiskLevel, ToolBlockedError

logger = logging.getLogger("datops_agent")


def wrap_crewai_tool(tool: Any, trust_gate, risk_level: Union[str, RiskLevel] = RiskLevel.MEDIUM) -> Any:
    """
    Wrap a single CrewAI tool with trust gate enforcement.

    Intercepts the tool's _run method to add pre/post checks.
    """
    if isinstance(risk_level, str):
        try:
            risk_level = RiskLevel(risk_level.lower())
        except ValueError:
            risk_level = RiskLevel.MEDIUM

    tool_name = getattr(tool, "name", "crewai_tool")
    original_run = tool._run if hasattr(tool, "_run") else None

    if original_run is None:
        logger.warning(f"CrewAI tool '{tool_name}' has no _run method, skipping wrap")
        return tool

    def wrapped_run(*args: Any, **kwargs: Any) -> Any:
        # Pre-execution gate
        decision = trust_gate.pre_tool_call(tool_name, kwargs or {}, risk_level)

        if not decision.allowed:
            raise ToolBlockedError(
                reason=decision.reason,
                trust_score=decision.trust_score,
                sandbox_level=decision.sandbox_level,
            )

        # Execute with timing
        start = time.monotonic()
        try:
            result = original_run(*args, **kwargs)
            elapsed_ms = (time.monotonic() - start) * 1000

            trust_gate.post_tool_call(
                tool_name=tool_name,
                success=True,
                response_time_ms=elapsed_ms,
            )

            return result

        except ToolBlockedError:
            raise

        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000

            trust_gate.post_tool_call(
                tool_name=tool_name,
                success=False,
                response_time_ms=elapsed_ms,
                error=str(e),
            )

            raise

    tool._run = wrapped_run
    logger.debug(f"CrewAI tool '{tool_name}' wrapped with trust enforcement")
    return tool


def wrap_crewai(
    crew: Any,
    trust_gate,
    tool_risk_levels: Optional[Dict[str, Union[str, RiskLevel]]] = None,
) -> Any:
    """
    Wrap a CrewAI Crew with DatOps trust enforcement.

    Wraps all tools on all agents in the crew.

    Args:
        crew: A CrewAI Crew instance
        trust_gate: TrustGate instance
        tool_risk_levels: Optional dict of tool_name → risk level

    Returns:
        The crew with all tools wrapped
    """
    tool_risk_levels = tool_risk_levels or {}
    wrapped_count = 0

    # Wrap tools on each agent in the crew
    agents: List[Any] = []
    if hasattr(crew, "agents"):
        agents = crew.agents or []

    for agent in agents:
        if not hasattr(agent, "tools"):
            continue

        for i, tool in enumerate(agent.tools):
            tool_name = getattr(tool, "name", f"tool_{i}")
            risk = tool_risk_levels.get(tool_name, RiskLevel.MEDIUM)
            agent.tools[i] = wrap_crewai_tool(tool, trust_gate, risk)
            wrapped_count += 1

    # Also wrap any crew-level tools
    if hasattr(crew, "tools"):
        for i, tool in enumerate(crew.tools or []):
            tool_name = getattr(tool, "name", f"crew_tool_{i}")
            risk = tool_risk_levels.get(tool_name, RiskLevel.MEDIUM)
            crew.tools[i] = wrap_crewai_tool(tool, trust_gate, risk)
            wrapped_count += 1

    logger.info(f"CrewAI crew wrapped: {wrapped_count} tools with trust enforcement")
    return crew
