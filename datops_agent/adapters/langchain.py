"""
DatOps Agent SDK — LangChain adapter

Hooks into LangChain's callback system to enforce trust-gated tool execution.
Injects a CallbackHandler that intercepts on_tool_start/end/error.

Usage:
    from datops_agent import DatOps
    agent = DatOps.wrap_langchain(my_agent, api_key="dat_xxx")
"""

import logging
import time
from typing import Any, Dict, Optional, Union

from ..types import RiskLevel, ToolBlockedError

logger = logging.getLogger("datops_agent")


def _get_langchain_base():
    """Import LangChain base callback handler."""
    try:
        from langchain_core.callbacks import BaseCallbackHandler
        return BaseCallbackHandler
    except ImportError:
        try:
            from langchain.callbacks.base import BaseCallbackHandler
            return BaseCallbackHandler
        except ImportError:
            raise ImportError(
                "LangChain is required for this adapter. "
                "Install it with: pip install datops-agent-sdk[langchain]"
            )


class DatOpsCallbackHandler:
    """
    LangChain CallbackHandler for trust-gated tool execution.

    Hooks into:
    - on_tool_start → pre_tool_call() → raise ToolBlockedError if denied
    - on_tool_end → post_tool_call(success=True)
    - on_tool_error → post_tool_call(success=False)
    """

    def __init__(self, trust_gate, tool_risk_levels: Optional[Dict[str, str]] = None):
        # Dynamically inherit from LangChain's BaseCallbackHandler
        BaseCallbackHandler = _get_langchain_base()
        self.__class__ = type(
            "DatOpsCallbackHandler",
            (BaseCallbackHandler,),
            {
                "on_tool_start": self._on_tool_start,
                "on_tool_end": self._on_tool_end,
                "on_tool_error": self._on_tool_error,
            },
        )
        self._gate = trust_gate
        self._tool_risk_levels = tool_risk_levels or {}
        self._tool_start_times: Dict[str, float] = {}

    def _get_risk_level(self, tool_name: str) -> RiskLevel:
        """Get risk level for a tool, defaulting to MEDIUM."""
        level = self._tool_risk_levels.get(tool_name, "medium")
        try:
            return RiskLevel(level.lower()) if isinstance(level, str) else level
        except ValueError:
            return RiskLevel.MEDIUM

    def _on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Any = None,
        metadata: Any = None,
        inputs: Any = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts. Blocks execution if trust check fails."""
        tool_name = serialized.get("name", "unknown_tool")
        risk_level = self._get_risk_level(tool_name)

        # Track start time for response time measurement
        self._tool_start_times[tool_name] = time.monotonic()

        # Pre-execution gate
        decision = self._gate.pre_tool_call(tool_name, {}, risk_level)

        if not decision.allowed:
            raise ToolBlockedError(
                reason=decision.reason,
                trust_score=decision.trust_score,
                sandbox_level=decision.sandbox_level,
            )

    def _on_tool_end(
        self,
        output: str,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool completes successfully."""
        # We don't have tool_name here in the standard callback,
        # so we report for the most recent tool
        tool_name = "langchain_tool"
        elapsed_ms = None

        if self._tool_start_times:
            # Pop the most recent start time
            last_tool = list(self._tool_start_times.keys())[-1]
            start = self._tool_start_times.pop(last_tool, None)
            if start:
                elapsed_ms = (time.monotonic() - start) * 1000
            tool_name = last_tool

        self._gate.post_tool_call(
            tool_name=tool_name,
            success=True,
            response_time_ms=elapsed_ms,
        )

    def _on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool fails."""
        tool_name = "langchain_tool"
        elapsed_ms = None

        if self._tool_start_times:
            last_tool = list(self._tool_start_times.keys())[-1]
            start = self._tool_start_times.pop(last_tool, None)
            if start:
                elapsed_ms = (time.monotonic() - start) * 1000
            tool_name = last_tool

        self._gate.post_tool_call(
            tool_name=tool_name,
            success=False,
            response_time_ms=elapsed_ms,
            error=str(error),
        )


def wrap_langchain(
    agent_or_executor: Any,
    trust_gate,
    tool_risk_levels: Optional[Dict[str, str]] = None,
) -> Any:
    """
    Wrap a LangChain agent or AgentExecutor with DatOps trust enforcement.

    Injects a DatOpsCallbackHandler into the agent's callbacks.

    Args:
        agent_or_executor: A LangChain Agent or AgentExecutor
        trust_gate: TrustGate instance
        tool_risk_levels: Optional dict of tool_name → risk level

    Returns:
        The agent with DatOps callback handler injected
    """
    handler = DatOpsCallbackHandler(trust_gate, tool_risk_levels)

    # Try different callback injection patterns
    if hasattr(agent_or_executor, "callbacks"):
        if agent_or_executor.callbacks is None:
            agent_or_executor.callbacks = [handler]
        else:
            agent_or_executor.callbacks.append(handler)
    elif hasattr(agent_or_executor, "callback_manager"):
        if hasattr(agent_or_executor.callback_manager, "add_handler"):
            agent_or_executor.callback_manager.add_handler(handler)
    else:
        logger.warning(
            "Could not inject DatOps callback handler. "
            "Agent type may not support callbacks. "
            "Consider using the generic decorator adapter instead."
        )

    logger.info("LangChain agent wrapped with DatOps trust enforcement")
    return agent_or_executor
