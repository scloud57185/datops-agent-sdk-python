"""
DatOps Agent SDK — OpenAI SDK adapter

Wraps OpenAI Agents SDK tool functions with trust-gated execution.

Usage:
    from datops_agent import DatOps
    agent = DatOps.wrap_openai(my_agent, api_key="dat_xxx")
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union

from ..types import RiskLevel, ToolBlockedError

logger = logging.getLogger("datops_agent")


def wrap_openai_tool(
    tool: Any,
    trust_gate,
    risk_level: Union[str, RiskLevel] = RiskLevel.MEDIUM,
) -> Any:
    """
    Wrap a single OpenAI SDK tool function with trust gate.

    Supports both:
    - OpenAI Agents SDK `FunctionTool` objects (has `on_invoke_tool`)
    - Plain function tools (callable with __name__)
    """
    if isinstance(risk_level, str):
        try:
            risk_level = RiskLevel(risk_level.lower())
        except ValueError:
            risk_level = RiskLevel.MEDIUM

    # Handle FunctionTool objects (OpenAI Agents SDK)
    if hasattr(tool, "on_invoke_tool"):
        tool_name = getattr(tool, "name", "openai_tool")
        original_invoke = tool.on_invoke_tool

        async def wrapped_invoke(ctx: Any, args: str) -> str:
            # Pre-execution gate
            decision = trust_gate.pre_tool_call(tool_name, {"args": args}, risk_level)
            if not decision.allowed:
                raise ToolBlockedError(
                    reason=decision.reason,
                    trust_score=decision.trust_score,
                    sandbox_level=decision.sandbox_level,
                )

            start = time.monotonic()
            try:
                result = await original_invoke(ctx, args)
                elapsed_ms = (time.monotonic() - start) * 1000
                trust_gate.post_tool_call(tool_name, True, elapsed_ms)
                return result
            except ToolBlockedError:
                raise
            except Exception as e:
                elapsed_ms = (time.monotonic() - start) * 1000
                trust_gate.post_tool_call(tool_name, False, elapsed_ms, str(e))
                raise

        tool.on_invoke_tool = wrapped_invoke
        logger.debug(f"OpenAI tool '{tool_name}' wrapped (FunctionTool)")
        return tool

    # Handle plain callable tools
    if callable(tool):
        tool_name = getattr(tool, "__name__", "openai_tool")
        return trust_gate.wrap_tool(tool, tool_name, risk_level)

    logger.warning(f"Unsupported OpenAI tool type: {type(tool)}")
    return tool


def wrap_openai(
    agent: Any,
    trust_gate,
    tool_risk_levels: Optional[Dict[str, Union[str, RiskLevel]]] = None,
) -> Any:
    """
    Wrap an OpenAI Agents SDK agent with DatOps trust enforcement.

    Wraps all tool functions on the agent.

    Args:
        agent: An OpenAI Agent instance
        trust_gate: TrustGate instance
        tool_risk_levels: Optional dict of tool_name → risk level

    Returns:
        The agent with all tools wrapped
    """
    tool_risk_levels = tool_risk_levels or {}
    wrapped_count = 0

    # Wrap tools list
    if hasattr(agent, "tools"):
        tools = agent.tools or []
        for i, tool in enumerate(tools):
            tool_name = getattr(tool, "name", getattr(tool, "__name__", f"tool_{i}"))
            risk = tool_risk_levels.get(tool_name, RiskLevel.MEDIUM)
            tools[i] = wrap_openai_tool(tool, trust_gate, risk)
            wrapped_count += 1

    logger.info(f"OpenAI agent wrapped: {wrapped_count} tools with trust enforcement")
    return agent


def wrap_tool_list(
    tools: List[Any],
    trust_gate,
    tool_risk_levels: Optional[Dict[str, Union[str, RiskLevel]]] = None,
) -> List[Any]:
    """
    Wrap a list of OpenAI tool functions.

    Useful when tools are passed directly to Runner.run() rather
    than attached to an Agent.

    Args:
        tools: List of tool functions or FunctionTool objects
        trust_gate: TrustGate instance
        tool_risk_levels: Optional dict of tool_name → risk level

    Returns:
        List of wrapped tools
    """
    tool_risk_levels = tool_risk_levels or {}
    wrapped = []
    for tool in tools:
        tool_name = getattr(tool, "name", getattr(tool, "__name__", "tool"))
        risk = tool_risk_levels.get(tool_name, RiskLevel.MEDIUM)
        wrapped.append(wrap_openai_tool(tool, trust_gate, risk))
    return wrapped
