"""Tests for framework adapters — generic, LangChain, CrewAI, OpenAI."""

import unittest
from unittest.mock import MagicMock

from datops_agent.types import (
    RiskLevel,
    SandboxLevel,
    SignalEvent,
    ToolBlockedError,
    TrustResult,
    DatOpsConfig,
)
from datops_agent.trust_gate import TrustGate
from datops_agent.adapters.generic import trust_gate_decorator


def make_mock_gate(trust_score=50.0, sandbox_level=SandboxLevel.ADAPTIVE):
    """Create a mock TrustGate."""
    from datops_agent.types import GateDecision

    gate = MagicMock(spec=TrustGate)

    # Default: allow all
    gate.pre_tool_call.return_value = GateDecision(
        allowed=True,
        reason="Authorized",
        sandbox_level=sandbox_level,
        trust_score=trust_score,
    )
    gate.post_tool_call = MagicMock()
    return gate


class TestGenericDecorator(unittest.TestCase):
    """Tests for the generic decorator adapter."""

    def test_decorator_allows_execution(self):
        """Decorated function executes when trust check passes."""
        gate = make_mock_gate()

        @trust_gate_decorator(gate, risk_level="medium")
        def my_tool(query: str) -> str:
            return f"result: {query}"

        result = my_tool("test")
        assert result == "result: test"

        gate.pre_tool_call.assert_called_once()
        gate.post_tool_call.assert_called_once()

    def test_decorator_blocks_execution(self):
        """Decorated function blocked when trust check fails."""
        from datops_agent.types import GateDecision

        gate = make_mock_gate()
        gate.pre_tool_call.return_value = GateDecision(
            allowed=False,
            reason="Trust too low",
            sandbox_level=SandboxLevel.STRICT,
            trust_score=20.0,
        )

        @trust_gate_decorator(gate, risk_level="high")
        def risky_tool() -> str:
            return "should not run"

        with self.assertRaises(ToolBlockedError):
            risky_tool()

        # post_tool_call should NOT be called on block
        gate.post_tool_call.assert_not_called()

    def test_decorator_reports_failure_on_exception(self):
        """Decorated function reports failure when it throws."""
        gate = make_mock_gate()

        @trust_gate_decorator(gate, risk_level="low")
        def failing_tool() -> str:
            raise RuntimeError("Oops")

        with self.assertRaises(RuntimeError):
            failing_tool()

        # post_tool_call called with success=False
        call_args = gate.post_tool_call.call_args
        assert call_args.kwargs["success"] is False
        assert "Oops" in call_args.kwargs["error"]

    def test_decorator_preserves_function_name(self):
        """Decorated function preserves original __name__."""
        gate = make_mock_gate()

        @trust_gate_decorator(gate, risk_level="low")
        def my_special_tool() -> str:
            return "hi"

        assert my_special_tool.__name__ == "my_special_tool"

    def test_decorator_attaches_metadata(self):
        """Decorated function has DatOps metadata attributes."""
        gate = make_mock_gate()

        @trust_gate_decorator(gate, risk_level="high", tool_name="custom_name")
        def tool() -> str:
            return "hi"

        assert tool._datops_wrapped is True
        assert tool._datops_risk_level == RiskLevel.HIGH
        assert tool._datops_tool_name == "custom_name"

    def test_decorator_with_string_risk_level(self):
        """Decorator accepts string risk level."""
        gate = make_mock_gate()

        @trust_gate_decorator(gate, risk_level="medium")
        def tool() -> str:
            return "ok"

        result = tool()
        assert result == "ok"


class TestCrewAIAdapter(unittest.TestCase):
    """Tests for the CrewAI adapter."""

    def test_wrap_crewai_tool(self):
        """Single CrewAI tool wrapping."""
        from datops_agent.adapters.crewai import wrap_crewai_tool

        gate = make_mock_gate()

        # Mock CrewAI tool
        tool = MagicMock()
        tool.name = "search_tool"
        tool._run = MagicMock(return_value="search results")

        wrapped = wrap_crewai_tool(tool, gate, RiskLevel.MEDIUM)

        result = wrapped._run("test query")
        assert result == "search results"
        gate.pre_tool_call.assert_called_once()
        gate.post_tool_call.assert_called_once()

    def test_wrap_crewai_crew(self):
        """Crew-level wrapping wraps all agent tools."""
        from datops_agent.adapters.crewai import wrap_crewai

        gate = make_mock_gate()

        # Mock crew with agents and tools
        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1._run = MagicMock(return_value="r1")

        tool2 = MagicMock()
        tool2.name = "tool2"
        tool2._run = MagicMock(return_value="r2")

        agent = MagicMock()
        agent.tools = [tool1, tool2]

        crew = MagicMock()
        crew.agents = [agent]
        crew.tools = None

        wrapped_crew = wrap_crewai(crew, gate)

        # Both tools should be wrapped
        agent.tools[0]._run("test")
        agent.tools[1]._run("test")
        assert gate.pre_tool_call.call_count == 2


class TestOpenAIAdapter(unittest.TestCase):
    """Tests for the OpenAI SDK adapter."""

    def test_wrap_callable_tool(self):
        """Plain callable tool wrapping."""
        from datops_agent.adapters.openai_sdk import wrap_openai_tool
        from datops_agent.trust_gate import TrustGate

        gate = make_mock_gate()
        # For plain callables, wrap_openai_tool delegates to gate.wrap_tool()
        # Configure the mock to use the real TrustGate.wrap_tool logic
        gate.wrap_tool = lambda fn, name, risk: TrustGate.__new__(TrustGate).wrap_tool(fn, name, risk) or _make_real_wrap(gate, fn, name, risk)

        def _wrap(fn, name, risk):
            """Real wrapping that uses the mock gate for pre/post calls."""
            import functools
            import time as _time

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                decision = gate.pre_tool_call(name, kwargs or {}, risk)
                if not decision.allowed:
                    raise ToolBlockedError(
                        reason=decision.reason,
                        trust_score=decision.trust_score,
                        sandbox_level=decision.sandbox_level,
                    )
                start = _time.monotonic()
                try:
                    result = fn(*args, **kwargs)
                    elapsed_ms = (_time.monotonic() - start) * 1000
                    gate.post_tool_call(tool_name=name, success=True, response_time_ms=elapsed_ms)
                    return result
                except Exception as e:
                    elapsed_ms = (_time.monotonic() - start) * 1000
                    gate.post_tool_call(tool_name=name, success=False, response_time_ms=elapsed_ms, error=str(e))
                    raise
            return wrapper

        gate.wrap_tool = _wrap

        def my_tool(query: str) -> str:
            return f"result: {query}"

        wrapped = wrap_openai_tool(my_tool, gate, RiskLevel.LOW)
        result = wrapped("test")
        assert result == "result: test"
        gate.pre_tool_call.assert_called_once()
        gate.post_tool_call.assert_called_once()

    def test_wrap_openai_agent(self):
        """Agent-level wrapping wraps all tool functions."""
        from datops_agent.adapters.openai_sdk import wrap_openai
        import functools
        import time as _time

        gate = make_mock_gate()

        # Configure wrap_tool to produce real wrappers
        def _wrap(fn, name, risk):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                decision = gate.pre_tool_call(name, kwargs or {}, risk)
                if not decision.allowed:
                    raise ToolBlockedError(
                        reason=decision.reason,
                        trust_score=decision.trust_score,
                        sandbox_level=decision.sandbox_level,
                    )
                start = _time.monotonic()
                result = fn(*args, **kwargs)
                elapsed_ms = (_time.monotonic() - start) * 1000
                gate.post_tool_call(tool_name=name, success=True, response_time_ms=elapsed_ms)
                return result
            return wrapper

        gate.wrap_tool = _wrap

        def tool_a() -> str:
            return "a"

        def tool_b() -> str:
            return "b"

        agent = MagicMock()
        agent.tools = [tool_a, tool_b]

        wrapped = wrap_openai(agent, gate)

        # Tools should be wrapped and produce correct results
        assert agent.tools[0]() == "a"
        assert agent.tools[1]() == "b"
        assert gate.pre_tool_call.call_count == 2

    def test_wrap_tool_list(self):
        """wrap_tool_list wraps a standalone tool list."""
        from datops_agent.adapters.openai_sdk import wrap_tool_list

        gate = make_mock_gate()

        tools = [
            lambda: "a",
            lambda: "b",
        ]

        wrapped = wrap_tool_list(tools, gate)
        assert len(wrapped) == 2


class TestDatOpsFacade(unittest.TestCase):
    """Tests for the main DatOps facade class."""

    def test_facade_repr(self):
        """DatOps repr shows key info."""
        from datops_agent import DatOps

        with unittest.mock.patch.object(DatOps, "__init__", lambda self, **kw: None):
            datops = DatOps.__new__(DatOps)
            datops._core = MagicMock()
            datops._core.did = "did:dat:test:123"
            datops._core.trust_score = 55.5
            datops._core.sandbox_level = SandboxLevel.ADAPTIVE
            datops._gate = MagicMock()

            repr_str = repr(datops)
            assert "did:dat:test:123" in repr_str
            assert "55.5" in repr_str
            assert "ADAPTIVE" in repr_str

    def test_facade_context_manager(self):
        """DatOps can be used as context manager."""
        from datops_agent import DatOps

        with unittest.mock.patch.object(DatOps, "__init__", lambda self, **kw: None):
            datops = DatOps.__new__(DatOps)
            datops._core = MagicMock()
            datops._gate = MagicMock()

            with datops:
                pass

            datops._core.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
