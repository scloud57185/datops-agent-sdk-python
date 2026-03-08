"""
DatOps Agent SDK — Core engine

Handles agent registration, trust score management, signal reporting,
and heartbeat. Framework-agnostic — adapters build on top of this.

Mirrors: src/services/dat-agent/src/services/datAgentIdentity.ts
"""

import json
import logging
import os
import stat
import time
from typing import Optional, Dict, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .types import (
    DatOpsConfig,
    AgentIdentity,
    TrustResult,
    SandboxLevel,
    SignalEvent,
    get_sandbox_level,
    RegistrationError,
    DatOpsError,
)
from .cache import TrustCache
from .heartbeat import HeartbeatWorker

logger = logging.getLogger("datops_agent")


class DatOpsCore:
    """
    Core DAT trust enforcement engine. Framework-agnostic.

    Handles:
    - Auto-registration with DAT platform
    - Trust score caching (60s TTL, no Redis)
    - Signal reporting (fire-and-forget)
    - Background heartbeat for activity watchdog
    - Identity persistence across restarts
    """

    def __init__(self, config: DatOpsConfig):
        self._config = config
        self._identity: Optional[AgentIdentity] = None
        self._cache = TrustCache(default_ttl=config.trust_cache_ttl)
        self._heartbeat: Optional[HeartbeatWorker] = None
        self._session = self._create_session()
        self._initialized = False

        if config.debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry logic."""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _url(self, path: str) -> str:
        """Build full URL from path, using SDK gateway prefix."""
        base = self._config.base_url.rstrip("/")
        prefix = self._config.sdk_path_prefix.rstrip("/")
        return f"{base}{prefix}{path}"

    def _headers(self) -> Dict[str, str]:
        """Standard request headers."""
        headers = {"Content-Type": "application/json"}
        if self._identity and self._identity.api_key:
            headers["Authorization"] = f"Bearer {self._identity.api_key}"
        return headers

    # ========================================================================
    # Initialization
    # ========================================================================

    def initialize(self) -> AgentIdentity:
        """
        Auto-registration flow:
        1. Check persisted identity file
        2. Register agent with DAT platform
        3. Initialize reputation at 50
        4. Self-verify agent
        5. Report verification event
        6. Start heartbeat worker

        Returns AgentIdentity with DID, trust score, sandbox level.
        """
        if self._initialized and self._identity:
            return self._identity

        # Step 1: Check for persisted identity
        if self._config.persist_identity:
            loaded = self._load_persisted_identity()
            if loaded:
                self._identity = loaded
                self._initialized = True
                self._refresh_trust_score()
                self._start_heartbeat()
                logger.info(f"Loaded persisted identity: {self._identity.did}")
                return self._identity

        if not self._config.auto_register:
            raise RegistrationError("auto_register=False and no persisted identity found")

        # Step 2: Register agent
        self._register_agent()

        # Step 3: Initialize reputation
        self._initialize_reputation()

        # Step 4: Self-verify
        self._verify_agent()

        # Step 5: Report verification event
        self.report_signal(
            SignalEvent.VERIFICATION,
            details={
                "action": "agent_verification",
                "level": "basic",
                "source": "datops_agent_sdk",
            },
        )

        # Refresh trust score after verification
        self._refresh_trust_score()

        # Step 6: Persist identity
        if self._config.persist_identity:
            self._persist_identity()

        # Step 7: Start heartbeat
        self._start_heartbeat()

        self._initialized = True
        logger.info(
            f"Agent initialized: did={self._identity.did}, "
            f"trust={self._identity.trust_score:.1f}, "
            f"sandbox={self._identity.sandbox_level.value}"
        )
        return self._identity

    def _register_agent(self) -> None:
        """Register agent with DAT platform. Retries with exponential backoff."""
        agent_name = self._config.agent_name or f"datops_sdk_{int(time.time())}"

        body = {
            "name": agent_name,
            "description": self._config.agent_description,
            "version": "1.0.0",
            "providerOrgDid": self._config.org_did or f"did:dat:{self._config.network}:org_sdk_default",
            "capabilities": {
                "protocols": ["dat/v1"],
                "actions": [
                    {"name": "tool_call", "description": "Execute tools with trust enforcement", "riskLevel": "medium"},
                ],
                "dataAccess": ["sdk-managed"],
                "externalServices": ["llm-provider"],
            },
            "endpoints": {
                "primary": "https://sdk.datops.ai/local",
                "healthCheck": "https://sdk.datops.ai/health",
            },
        }

        last_error = None
        for attempt in range(1, self._config.retry_attempts + 1):
            try:
                logger.debug(f"Registering agent (attempt {attempt}/{self._config.retry_attempts})")
                resp = self._session.post(
                    self._url("/agents"),
                    json=body,
                    headers={"Content-Type": "application/json"},
                    timeout=self._config.request_timeout,
                )
                data = resp.json()

                if data.get("success"):
                    self._identity = AgentIdentity(
                        did=data["data"]["agentDid"],
                        api_key=data["data"].get("apiKey", ""),
                        trust_score=0,
                        sandbox_level=SandboxLevel.STRICT,
                        verification_status="pending",
                        registered_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    )
                    logger.info(f"Agent registered: {self._identity.did}")
                    return
                else:
                    last_error = data.get("error", str(data))
                    logger.warning(f"Registration attempt {attempt} failed: {last_error}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Registration attempt {attempt} error: {e}")

            if attempt < self._config.retry_attempts:
                delay = self._config.retry_base_delay * attempt
                logger.debug(f"Retrying in {delay}s...")
                time.sleep(delay)

        raise RegistrationError(f"All {self._config.retry_attempts} registration attempts failed: {last_error}")

    def _initialize_reputation(self) -> None:
        """Initialize reputation score at 50."""
        if not self._identity:
            return
        try:
            self._session.post(
                self._url(f"/reputation/{self._identity.did}/initialize"),
                json={},
                headers=self._headers(),
                timeout=self._config.signal_timeout,
            )
            logger.debug("Reputation initialized")
        except Exception as e:
            logger.debug(f"Reputation init (best effort): {e}")

    def _verify_agent(self) -> None:
        """Self-verify via org admin."""
        if not self._identity:
            return
        try:
            resp = self._session.post(
                self._url(f"/verification/agents/{self._identity.did}/verify-direct"),
                json={
                    "orgDid": self._config.org_did or f"did:dat:{self._config.network}:org_sdk_default",
                    "level": "basic",
                    "notes": "Auto-verified via DatOps Agent SDK",
                },
                headers=self._headers(),
                timeout=self._config.request_timeout,
            )
            data = resp.json()
            if data.get("success"):
                self._identity.verification_status = "verified"
                logger.debug("Agent verified")
        except Exception as e:
            # May already be verified
            if self._identity:
                self._identity.verification_status = "verified"
            logger.debug(f"Agent verification (best effort): {e}")

    # ========================================================================
    # Trust Score
    # ========================================================================

    def get_trust_score(self, force_refresh: bool = False) -> TrustResult:
        """
        Get trust score. Uses cache (60s TTL) unless force_refresh=True.
        Falls back to HTTP GET on cache miss.
        """
        if not force_refresh:
            cached = self._cache.get("trust_score")
            if cached is not None:
                return cached

        return self._refresh_trust_score()

    def _refresh_trust_score(self) -> TrustResult:
        """Fetch trust score from reputation service."""
        if not self._identity:
            result = TrustResult(trust_score=0, sandbox_level=SandboxLevel.STRICT)
            return result

        try:
            resp = self._session.get(
                self._url(f"/reputation/{self._identity.did}"),
                headers=self._headers(),
                timeout=self._config.signal_timeout,
            )
            data = resp.json()

            # Handle both { trustScore } and { data: { trustScore } } formats
            score_data = data.get("data", data)
            trust_score = float(score_data.get("trustScore", 0))
            sandbox_level = get_sandbox_level(trust_score)

            result = TrustResult(
                trust_score=trust_score,
                sandbox_level=sandbox_level,
                reliability_score=float(score_data.get("reliabilityScore", 50)),
                performance_score=float(score_data.get("performanceScore", 50)),
                compliance_score=float(score_data.get("complianceScore", 50)),
                security_score=float(score_data.get("securityScore", 50)),
                reporting_fidelity_score=float(score_data.get("reportingFidelityScore", 50)),
                investigation_state=score_data.get("investigationState", "active"),
            )

            # Update identity
            self._identity.trust_score = trust_score
            self._identity.sandbox_level = sandbox_level
            self._identity.total_interactions = int(score_data.get("totalInteractions", 0))

            # Cache the result
            self._cache.set("trust_score", result, self._config.trust_cache_ttl)

            return result

        except Exception as e:
            logger.debug(f"Trust score refresh failed (using last known): {e}")
            # Return last known or defaults
            if self._identity:
                return TrustResult(
                    trust_score=self._identity.trust_score,
                    sandbox_level=self._identity.sandbox_level,
                )
            return TrustResult(trust_score=0, sandbox_level=SandboxLevel.STRICT)

    # ========================================================================
    # Signal Reporting
    # ========================================================================

    def report_signal(
        self,
        event: SignalEvent,
        response_time_ms: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Report a trust signal to the reputation service.
        Fire-and-forget — never blocks, never raises.
        """
        if not self._identity or "offline" in self._identity.did:
            return

        body: Dict[str, Any] = {"event": event.value if isinstance(event, SignalEvent) else event}

        if response_time_ms is not None:
            body["responseTime"] = max(1, min(response_time_ms, 60000))

        if details:
            body["details"] = details

        try:
            resp = self._session.post(
                self._url(f"/reputation/{self._identity.did}/update"),
                json=body,
                headers=self._headers(),
                timeout=self._config.signal_timeout,
            )

            if resp.ok:
                data = resp.json()
                # Update local trust score from response
                new_score = data.get("trustScore")
                if new_score is not None and self._identity:
                    self._identity.trust_score = float(new_score)
                    self._identity.sandbox_level = get_sandbox_level(float(new_score))
                    # Invalidate cache so next read gets fresh value
                    self._cache.delete("trust_score")
        except Exception as e:
            logger.debug(f"Signal report failed (best effort): {e}")

    # ========================================================================
    # Heartbeat
    # ========================================================================

    def _send_heartbeat(self) -> None:
        """Send heartbeat to reputation service."""
        if not self._identity or "offline" in self._identity.did:
            return
        try:
            self._session.post(
                self._url(f"/reputation/{self._identity.did}/heartbeat"),
                json={},
                headers=self._headers(),
                timeout=self._config.signal_timeout,
            )
        except Exception:
            pass  # Fire-and-forget

    def _start_heartbeat(self) -> None:
        """Start background heartbeat worker."""
        if self._heartbeat and self._heartbeat.is_alive:
            return
        self._heartbeat = HeartbeatWorker(
            heartbeat_fn=self._send_heartbeat,
            interval=self._config.heartbeat_interval,
        )
        self._heartbeat.start()

    # ========================================================================
    # Identity Persistence
    # ========================================================================

    def _persist_identity(self) -> None:
        """Save identity to JSON file for restart survival."""
        if not self._config.persist_identity or not self._identity:
            return
        try:
            data = {
                "did": self._identity.did,
                "api_key": self._identity.api_key,
                "trust_score": self._identity.trust_score,
                "verification_status": self._identity.verification_status,
                "registered_at": self._identity.registered_at,
            }
            path = self._config.persist_identity
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            # Secure file permissions (owner read/write only)
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
            logger.debug(f"Identity persisted to {path}")
        except Exception as e:
            logger.warning(f"Failed to persist identity: {e}")

    def _load_persisted_identity(self) -> Optional[AgentIdentity]:
        """Load identity from JSON file."""
        path = self._config.persist_identity
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return AgentIdentity(
                did=data["did"],
                api_key=data.get("api_key", ""),
                trust_score=float(data.get("trust_score", 0)),
                sandbox_level=get_sandbox_level(float(data.get("trust_score", 0))),
                verification_status=data.get("verification_status", "verified"),
                registered_at=data.get("registered_at", ""),
            )
        except Exception as e:
            logger.warning(f"Failed to load persisted identity: {e}")
            return None

    # ========================================================================
    # Lifecycle
    # ========================================================================

    def shutdown(self) -> None:
        """Stop heartbeat and cleanup."""
        if self._heartbeat:
            self._heartbeat.stop()
        self._session.close()
        logger.debug("DatOpsCore shutdown complete")

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def did(self) -> str:
        return self._identity.did if self._identity else "not-registered"

    @property
    def trust_score(self) -> float:
        return self._identity.trust_score if self._identity else 0.0

    @property
    def sandbox_level(self) -> SandboxLevel:
        return self._identity.sandbox_level if self._identity else SandboxLevel.STRICT

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._identity is not None

    @property
    def identity(self) -> Optional[AgentIdentity]:
        return self._identity

    @property
    def config(self) -> DatOpsConfig:
        return self._config
