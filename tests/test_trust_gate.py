"""Tests for TrustGate — pre/post tool call middleware."""

import unittest
from unittest.mock import MagicMock, patch

from datops_agent.trust_gate import TrustGate
from datops_agent.types import (
    DatOpsConfig,
    RiskLevel,
    SandboxLevel,
    SignalEvent,
    ToolBlockedError,
    TrustResult,
)


def make_mock_core(trust_score=50.0, sandbox_level=SandboxLevel.ADAPTIVE):
    """Create a mock DatOpsCore with given trust/sandbox."""
    core = MagicMock()
    core.get_trust_score.return_value = TrustResult(
        trust_score=trust_score,
        sandbox_level=sandbox_level,
    )
    core.config = DatOpsConfig(
        api_key="test",
        min_trust_for_tool=0.0,
        trust_threshold_high_risk=70.0,
    )
    core.report_signal = MagicMock()
    return core


class TestPreToolCall(unittest.TestCase):
    """Tests for pre_tool_call authorization."""

    def test_low_risk_allowed_at_strict(self):
        """Low-risk tools allowed at STRICT sandbox level."""
        core = make_mock_core(trust_score=20.0, sandbox_level=SandboxLevel.STRICT)
        gate = TrustGate(core)

        decision = gate.pre_tool_call("read_file", {}, RiskLevel.LOW)
        assert decision.allowed is True

    def test_medium_risk_blocked_at_strict(self):
        """Medium-risk tools blocked at STRICT sandbox level."""
        core = make_mock_core(trust_score=20.0, sandbox_level=SandboxLevel.STRICT)
        gate = TrustGate(core)

        decision = gate.pre_tool_call("web_search", {}, RiskLevel.MEDIUM)
        assert decision.allowed is False
        assert "STRICT" in decision.reason

    def test_medium_risk_allowed_at_adaptive(self):
        """Medium-risk tools allowed at ADAPTIVE sandbox level."""
        core = make_mock_core(trust_score=50.0, sandbox_level=SandboxLevel.ADAPTIVE)
        gate = TrustGate(core)

        decision = gate.pre_tool_call("web_search", {}, RiskLevel.MEDIUM)
        assert decision.allowed is True

    def test_high_risk_blocked_at_adaptive(self):
        """High-risk tools blocked at ADAPTIVE sandbox level."""
        core = make_mock_core(trust_score=50.0, sandbox_level=SandboxLevel.ADAPTIVE)
        gate = TrustGate(core)

        decision = gate.pre_tool_call("send_email", {}, RiskLevel.HIGH)
        assert decision.allowed is False

    def test_high_risk_allowed_at_open(self):
        """High-risk tools allowed at OPEN sandbox level."""
        core = make_mock_core(trust_score=80.0, sandbox_level=SandboxLevel.OPEN)
        gate = TrustGate(core)

        decision = gate.pre_tool_call("send_email", {}, RiskLevel.HIGH)
        assert decision.allowed is True

    def test_high_risk_blocked_below_threshold(self):
        """High-risk tools blocked when trust below trust_threshold_high_risk."""
        core = make_mock_core(trust_score=65.0, sandbox_level=SandboxLevel.ADAPTIVE)
        # Even if sandbox somehow allows, threshold should block
        core.get_trust_score.return_value = TrustResult(
            trust_score=65.0,
            sandbox_level=SandboxLevel.OPEN,  # Pretend open
        )
        gate = TrustGate(core)

        decision = gate.pre_tool_call("send_email", {}, RiskLevel.HIGH)
        assert decision.allowed is False
        assert "70" in decision.reason  # threshold

    def test_string_risk_level_accepted(self):
        """String risk levels are converted to RiskLevel enum."""
        core = make_mock_core(trust_score=50.0, sandbox_level=SandboxLevel.ADAPTIVE)
        gate = TrustGate(core)

        decision = gate.pre_tool_call("tool", {}, "medium")
        assert decision.allowed is True

    def test_invalid_risk_level_defaults_medium(self):
        """Invalid risk level string defaults to MEDIUM."""
        core = make_mock_core(trust_score=50.0, sandbox_level=SandboxLevel.ADAPTIVE)
        gate = TrustGate(core)

        decision = gate.pre_tool_call("tool", {}, "extreme")
        assert decision.allowed is True  # medium allowed at ADAPTIVE

    def test_min_trust_for_tool_blocks(self):
        """min_trust_for_tool config blocks low-trust tools."""
        core = make_mock_core(trust_score=5.0, sandbox_level=SandboxLevel.STRICT)
        core.config.min_trust_for_tool = 10.0
        gate = TrustGate(core)

        decision = gate.pre_tool_call("read_file", {}, RiskLevel.LOW)
        assert decision.allowed is False
        assert "minimum threshold" in decision.reason


class TestPostToolCall(unittest.TestCase):
    """Tests for post_tool_call signal reporting."""

    def test_success_reports_action_success(self):
        """Successful tool call reports action_success signal."""
        core = make_mock_core()
        gate = TrustGate(core)

        gate.post_tool_call("web_search", success=True, response_time_ms=150.0)

        core.report_signal.assert_called_once()
        call_args = core.report_signal.call_args
        assert call_args.kwargs["event"] == SignalEvent.ACTION_SUCCESS
        assert call_args.kwargs["response_time_ms"] == 150.0
        assert call_args.kwargs["details"]["action"] == "web_search_ok"

    def test_failure_reports_action_failure(self):
        """Failed tool call reports action_failure signal with error."""
        core = make_mock_core()
        gate = TrustGate(core)

        gate.post_tool_call(
            "web_search",
            success=False,
            response_time_ms=200.0,
            error="Connection timeout",
        )

        call_args = core.report_signal.call_args
        assert call_args.kwargs["event"] == SignalEvent.ACTION_FAILURE
        assert call_args.kwargs["details"]["error"] == "Connection timeout"
        assert call_args.kwargs["details"]["action"] == "web_search_fail"

    def test_error_truncated(self):
        """Long error messages are truncated to 200 chars."""
        core = make_mock_core()
        gate = TrustGate(core)

        long_error = "x" * 500
        gate.post_tool_call("tool", success=False, error=long_error)

        call_args = core.report_signal.call_args
        assert len(call_args.kwargs["details"]["error"]) == 200


class TestWrapTool(unittest.TestCase):
    """Tests for wrap_tool functionality."""

    def test_wrap_tool_success(self):
        """Wrapped tool executes and reports success."""
        core = make_mock_core(trust_score=50.0, sandbox_level=SandboxLevel.ADAPTIVE)
        gate = TrustGate(core)

        def my_tool(query: str) -> str:
            return f"result: {query}"

        wrapped = gate.wrap_tool(my_tool, "my_tool", RiskLevel.MEDIUM)
        result = wrapped("test query")

        assert result == "result: test query"
        core.report_signal.assert_called_once()

    def test_wrap_tool_blocked(self):
        """Wrapped tool raises ToolBlockedError when trust check fails."""
        core = make_mock_core(trust_score=20.0, sandbox_level=SandboxLevel.STRICT)
        gate = TrustGate(core)

        def risky_tool() -> str:
            return "should not reach here"

        wrapped = gate.wrap_tool(risky_tool, "risky", RiskLevel.HIGH)

        with self.assertRaises(ToolBlockedError) as ctx:
            wrapped()
        assert ctx.exception.trust_score == 20.0
        assert ctx.exception.sandbox_level == SandboxLevel.STRICT

    def test_wrap_tool_error_reports_failure(self):
        """Wrapped tool reports failure signal on exception."""
        core = make_mock_core(trust_score=80.0, sandbox_level=SandboxLevel.OPEN)
        gate = TrustGate(core)

        def failing_tool() -> str:
            raise ValueError("Something broke")

        wrapped = gate.wrap_tool(failing_tool, "failing", RiskLevel.LOW)

        with self.assertRaises(ValueError):
            wrapped()

        call_args = core.report_signal.call_args
        assert call_args.kwargs["event"] == SignalEvent.ACTION_FAILURE
        assert "Something broke" in call_args.kwargs["details"]["error"]

    def test_wrap_tools_batch(self):
        """wrap_tools wraps multiple tools at once."""
        core = make_mock_core(trust_score=50.0, sandbox_level=SandboxLevel.ADAPTIVE)
        gate = TrustGate(core)

        tools = {
            "search": lambda q: f"found: {q}",
            "read": lambda p: f"content: {p}",
        }
        risk_levels = {"search": "medium", "read": "low"}

        wrapped = gate.wrap_tools(tools, risk_levels)

        assert wrapped["search"]("test") == "found: test"
        assert wrapped["read"]("/tmp") == "content: /tmp"


class TestGetCurrentSandbox(unittest.TestCase):
    """Tests for get_current_sandbox."""

    def test_returns_sandbox_info(self):
        core = make_mock_core(trust_score=50.0, sandbox_level=SandboxLevel.ADAPTIVE)
        gate = TrustGate(core)

        info = gate.get_current_sandbox()
        assert info["trust_score"] == 50.0
        assert info["sandbox_level"] == "ADAPTIVE"
        assert "low" in info["allowed_risk_levels"]
        assert "medium" in info["allowed_risk_levels"]
        assert "high" not in info["allowed_risk_levels"]


if __name__ == "__main__":
    unittest.main()
