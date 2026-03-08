"""
DatOps Agent SDK — Drop-in trust enforcement for AI agent frameworks.

2-line integration:

    from datops_agent import DatOps

    # LangChain
    agent = DatOps.wrap_langchain(my_agent, api_key="dat_xxx")

    # CrewAI
    crew = DatOps.wrap_crewai(my_crew, api_key="dat_xxx")

    # OpenAI Agents SDK
    agent = DatOps.wrap_openai(my_agent, api_key="dat_xxx")

    # Generic decorator
    datops = DatOps(api_key="dat_xxx")

    @datops.trust_gate(risk_level="medium")
    def my_tool(query: str) -> str:
        return search_web(query)
"""

__version__ = "0.1.0"

import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from .types import (
    DatOpsConfig,
    DatOpsError,
    AgentIdentity,
    GateDecision,
    RiskLevel,
    RegistrationError,
    SandboxLevel,
    SignalEvent,
    SignalReport,
    ToolBlockedError,
    TrustGateError,
    TrustResult,
    get_sandbox_level,
    SANDBOX_ALLOWED_RISKS,
)
from .core import DatOpsCore
from .trust_gate import TrustGate
from .cache import TrustCache
from .heartbeat import HeartbeatWorker

logger = logging.getLogger("datops_agent")

F = TypeVar("F", bound=Callable)


class DatOps:
    """
    DatOps Agent SDK — main facade.

    Provides trust-gated tool execution for any AI agent framework.

    Quick start:
        datops = DatOps(api_key="dat_xxx")
        print(f"Agent DID: {datops.did}")
        print(f"Trust Score: {datops.trust_score}")
        print(f"Sandbox: {datops.sandbox_level}")

    Framework wrappers:
        agent = DatOps.wrap_langchain(my_agent, api_key="dat_xxx")
        crew = DatOps.wrap_crewai(my_crew, api_key="dat_xxx")
        agent = DatOps.wrap_openai(my_agent, api_key="dat_xxx")

    Decorator:
        @datops.trust_gate(risk_level="medium")
        def my_tool(query: str) -> str: ...
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://www.datops.ai",
        agent_name: Optional[str] = None,
        agent_description: str = "Agent wrapped with DatOps trust enforcement",
        org_did: Optional[str] = None,
        network: str = "testnet",
        trust_cache_ttl: int = 60,
        heartbeat_interval: int = 300,
        min_trust_for_tool: float = 0.0,
        trust_threshold_high_risk: float = 70.0,
        persist_identity: Optional[str] = None,
        debug: bool = False,
        auto_register: bool = True,
        auto_initialize: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize DatOps SDK.

        Args:
            api_key: DatOps API key
            base_url: DatOps platform URL (default: https://www.datops.ai)
            agent_name: Name for this agent (auto-generated if not provided)
            agent_description: Description of this agent
            org_did: Organization DID (auto-generated if not provided)
            network: DAT network (testnet/mainnet)
            trust_cache_ttl: Trust score cache TTL in seconds (default: 60)
            heartbeat_interval: Heartbeat interval in seconds (default: 300)
            min_trust_for_tool: Minimum trust score to use any tool
            trust_threshold_high_risk: Minimum trust for high-risk tools (default: 70)
            persist_identity: Path to persist identity file (optional)
            debug: Enable debug logging
            auto_register: Auto-register agent on initialization
            auto_initialize: Initialize immediately (set False to call initialize() manually)
        """
        self._config = DatOpsConfig(
            api_key=api_key,
            base_url=base_url,
            agent_name=agent_name,
            agent_description=agent_description,
            org_did=org_did,
            network=network,
            trust_cache_ttl=trust_cache_ttl,
            heartbeat_interval=heartbeat_interval,
            min_trust_for_tool=min_trust_for_tool,
            trust_threshold_high_risk=trust_threshold_high_risk,
            persist_identity=persist_identity,
            debug=debug,
            auto_register=auto_register,
        )

        self._core = DatOpsCore(self._config)
        self._gate = TrustGate(self._core)

        if auto_initialize:
            self._core.initialize()

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def did(self) -> str:
        """Agent's Decentralized Identifier."""
        return self._core.did

    @property
    def trust_score(self) -> float:
        """Current trust score (0-100)."""
        return self._core.trust_score

    @property
    def sandbox_level(self) -> str:
        """Current sandbox level (STRICT/ADAPTIVE/OPEN)."""
        return self._core.sandbox_level.value

    @property
    def is_ready(self) -> bool:
        """Whether the agent is registered and initialized."""
        return self._core.is_ready

    @property
    def identity(self) -> Optional[AgentIdentity]:
        """Full agent identity."""
        return self._core.identity

    @property
    def core(self) -> DatOpsCore:
        """Access to the underlying core engine."""
        return self._core

    @property
    def gate(self) -> TrustGate:
        """Access to the trust gate middleware."""
        return self._gate

    # ========================================================================
    # Trust operations
    # ========================================================================

    def get_trust_score(self, force_refresh: bool = False) -> TrustResult:
        """Get current trust score with details."""
        return self._core.get_trust_score(force_refresh)

    def get_sandbox_info(self) -> Dict[str, Any]:
        """Get current sandbox level and allowed risk levels."""
        return self._gate.get_current_sandbox()

    def report_signal(
        self,
        event: Union[str, SignalEvent],
        response_time_ms: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Report a trust signal manually."""
        if isinstance(event, str):
            event = SignalEvent(event)
        self._core.report_signal(event, response_time_ms, details)

    # ========================================================================
    # Decorator pattern
    # ========================================================================

    def trust_gate(
        self,
        risk_level: Union[str, RiskLevel] = RiskLevel.MEDIUM,
        tool_name: Optional[str] = None,
    ) -> Callable[[F], F]:
        """
        Decorator for trust-gated function execution.

        Usage:
            @datops.trust_gate(risk_level="medium")
            def search_web(query: str) -> str:
                return requests.get(f"https://api.search.com?q={query}").text

            @datops.trust_gate(risk_level="high")
            def send_email(to: str, body: str) -> None:
                smtp.send(to, body)
        """
        from .adapters.generic import trust_gate_decorator
        return trust_gate_decorator(self._gate, risk_level, tool_name)

    # ========================================================================
    # Tool wrapping
    # ========================================================================

    def wrap_tool(
        self,
        fn: F,
        tool_name: Optional[str] = None,
        risk_level: Union[str, RiskLevel] = RiskLevel.MEDIUM,
    ) -> F:
        """Wrap a single callable with trust enforcement."""
        return self._gate.wrap_tool(fn, tool_name, risk_level)

    def wrap_tools(
        self,
        tools: Dict[str, Callable],
        risk_levels: Optional[Dict[str, Union[str, RiskLevel]]] = None,
        default_risk: Union[str, RiskLevel] = RiskLevel.MEDIUM,
    ) -> Dict[str, Callable]:
        """Wrap multiple callables with trust enforcement."""
        return self._gate.wrap_tools(tools, risk_levels, default_risk)

    # ========================================================================
    # Static framework wrappers (2-line integration)
    # ========================================================================

    @staticmethod
    def wrap_langchain(
        agent_or_executor: Any,
        api_key: str = "",
        base_url: str = "https://www.datops.ai",
        tool_risk_levels: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Wrap a LangChain agent with DatOps trust enforcement.

        Usage:
            from datops_agent import DatOps
            agent = DatOps.wrap_langchain(my_agent, api_key="dat_xxx")

        Args:
            agent_or_executor: LangChain Agent or AgentExecutor
            api_key: DatOps API key
            base_url: DatOps platform URL
            tool_risk_levels: Optional dict of tool_name → risk level
            **kwargs: Additional DatOps config options

        Returns:
            The agent with DatOps callback handler injected
        """
        datops = DatOps(api_key=api_key, base_url=base_url, **kwargs)

        from .adapters.langchain import wrap_langchain
        wrapped = wrap_langchain(agent_or_executor, datops._gate, tool_risk_levels)

        # Attach datops instance for later access
        wrapped._datops = datops  # type: ignore[attr-defined]
        return wrapped

    @staticmethod
    def wrap_crewai(
        crew: Any,
        api_key: str = "",
        base_url: str = "https://www.datops.ai",
        tool_risk_levels: Optional[Dict[str, Union[str, RiskLevel]]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Wrap a CrewAI Crew with DatOps trust enforcement.

        Usage:
            from datops_agent import DatOps
            crew = DatOps.wrap_crewai(my_crew, api_key="dat_xxx")

        Args:
            crew: CrewAI Crew instance
            api_key: DatOps API key
            base_url: DatOps platform URL
            tool_risk_levels: Optional dict of tool_name → risk level
            **kwargs: Additional DatOps config options

        Returns:
            The crew with all tools wrapped
        """
        datops = DatOps(api_key=api_key, base_url=base_url, **kwargs)

        from .adapters.crewai import wrap_crewai
        wrapped = wrap_crewai(crew, datops._gate, tool_risk_levels)

        wrapped._datops = datops  # type: ignore[attr-defined]
        return wrapped

    @staticmethod
    def wrap_openai(
        agent: Any,
        api_key: str = "",
        base_url: str = "https://www.datops.ai",
        tool_risk_levels: Optional[Dict[str, Union[str, RiskLevel]]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Wrap an OpenAI Agents SDK agent with DatOps trust enforcement.

        Usage:
            from datops_agent import DatOps
            agent = DatOps.wrap_openai(my_agent, api_key="dat_xxx")

        Args:
            agent: OpenAI Agent instance
            api_key: DatOps API key
            base_url: DatOps platform URL
            tool_risk_levels: Optional dict of tool_name → risk level
            **kwargs: Additional DatOps config options

        Returns:
            The agent with all tools wrapped
        """
        datops = DatOps(api_key=api_key, base_url=base_url, **kwargs)

        from .adapters.openai_sdk import wrap_openai
        wrapped = wrap_openai(agent, datops._gate, tool_risk_levels)

        wrapped._datops = datops  # type: ignore[attr-defined]
        return wrapped

    # ========================================================================
    # Lifecycle
    # ========================================================================

    def initialize(self) -> AgentIdentity:
        """Manually initialize (if auto_initialize=False)."""
        return self._core.initialize()

    def shutdown(self) -> None:
        """Stop heartbeat and cleanup resources."""
        self._core.shutdown()

    def __enter__(self) -> "DatOps":
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        return (
            f"DatOps(did={self.did!r}, "
            f"trust={self.trust_score:.1f}, "
            f"sandbox={self.sandbox_level!r})"
        )


# ============================================================================
# Public exports
# ============================================================================

__all__ = [
    # Main class
    "DatOps",
    # Core
    "DatOpsCore",
    "TrustGate",
    "TrustCache",
    "HeartbeatWorker",
    # Types
    "DatOpsConfig",
    "AgentIdentity",
    "TrustResult",
    "GateDecision",
    "SignalEvent",
    "SignalReport",
    "SandboxLevel",
    "RiskLevel",
    # Errors
    "DatOpsError",
    "RegistrationError",
    "ToolBlockedError",
    "TrustGateError",
    # Helpers
    "get_sandbox_level",
    "SANDBOX_ALLOWED_RISKS",
    # Version
    "__version__",
]
