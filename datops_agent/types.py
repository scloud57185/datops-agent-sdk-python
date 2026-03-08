"""
DatOps Agent SDK — Type definitions

Mirrors the sandbox level system from the DAT platform:
- STRICT (trust 0-30): Low-risk tools only
- ADAPTIVE (trust 30-70): Low + medium risk tools
- OPEN (trust 70-100): All tools including high-risk
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List


class SandboxLevel(str, Enum):
    """Trust-gated sandbox levels matching DAT platform."""
    STRICT = "STRICT"
    ADAPTIVE = "ADAPTIVE"
    OPEN = "OPEN"


class RiskLevel(str, Enum):
    """Tool risk classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SignalEvent(str, Enum):
    """Valid reputation signal event types."""
    ACTION_SUCCESS = "action_success"
    ACTION_FAILURE = "action_failure"
    ACTION_BLOCKED = "action_blocked"
    VERIFICATION = "verification"
    VIOLATION = "violation"
    FRAUD = "fraud"


def get_sandbox_level(trust_score: float) -> SandboxLevel:
    """Map trust score to sandbox level. Mirrors DAT platform logic."""
    if trust_score >= 70:
        return SandboxLevel.OPEN
    if trust_score >= 30:
        return SandboxLevel.ADAPTIVE
    return SandboxLevel.STRICT


# Which risk levels are allowed at each sandbox level
SANDBOX_ALLOWED_RISKS: Dict[SandboxLevel, List[RiskLevel]] = {
    SandboxLevel.STRICT: [RiskLevel.LOW],
    SandboxLevel.ADAPTIVE: [RiskLevel.LOW, RiskLevel.MEDIUM],
    SandboxLevel.OPEN: [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH],
}


@dataclass
class DatOpsConfig:
    """Configuration for DatOps Agent SDK."""
    api_key: str
    base_url: str = "https://www.datops.ai"
    agent_name: Optional[str] = None
    agent_description: str = "Agent wrapped with DatOps trust enforcement"
    org_did: Optional[str] = None
    network: str = "testnet"
    sdk_path_prefix: str = "/api/v1/sdk"
    trust_cache_ttl: int = 60
    heartbeat_interval: int = 300
    min_trust_for_tool: float = 0.0
    trust_threshold_high_risk: float = 70.0
    persist_identity: Optional[str] = None
    debug: bool = False
    auto_register: bool = True
    retry_attempts: int = 5
    retry_base_delay: float = 3.0
    request_timeout: float = 10.0
    signal_timeout: float = 3.0


@dataclass
class AgentIdentity:
    """Registered agent identity."""
    did: str
    api_key: str
    trust_score: float = 0.0
    sandbox_level: SandboxLevel = SandboxLevel.STRICT
    verification_status: str = "pending"
    total_interactions: int = 0
    registered_at: str = ""


@dataclass
class TrustResult:
    """Trust score query result."""
    trust_score: float
    sandbox_level: SandboxLevel
    reliability_score: float = 50.0
    performance_score: float = 50.0
    compliance_score: float = 50.0
    security_score: float = 50.0
    reporting_fidelity_score: float = 50.0
    investigation_state: str = "active"


@dataclass
class GateDecision:
    """Pre-execution authorization decision."""
    allowed: bool
    reason: str = ""
    sandbox_level: SandboxLevel = SandboxLevel.STRICT
    trust_score: float = 0.0


@dataclass
class SignalReport:
    """Signal to report to reputation service."""
    event: SignalEvent
    response_time_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


class DatOpsError(Exception):
    """Base exception for DatOps Agent SDK."""
    pass


class RegistrationError(DatOpsError):
    """Agent registration failed."""
    pass


class ToolBlockedError(DatOpsError):
    """Tool execution blocked by trust gate."""
    def __init__(self, reason: str, trust_score: float, sandbox_level: SandboxLevel):
        self.reason = reason
        self.trust_score = trust_score
        self.sandbox_level = sandbox_level
        super().__init__(
            f"Tool blocked: {reason} (trust={trust_score:.1f}, sandbox={sandbox_level.value})"
        )


class TrustGateError(DatOpsError):
    """Trust gate check failed."""
    pass
