"""Tests for DatOpsCore — registration, trust, signals."""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

import responses

from datops_agent.core import DatOpsCore
from datops_agent.types import (
    DatOpsConfig,
    AgentIdentity,
    SandboxLevel,
    SignalEvent,
    TrustResult,
    RegistrationError,
)


def make_config(**overrides):
    """Create a test config."""
    defaults = {
        "api_key": "test_key",
        "base_url": "http://localhost:3015",
        "agent_name": "test_agent",
        "auto_register": True,
        "retry_attempts": 2,
        "retry_base_delay": 0.01,
        "request_timeout": 2.0,
        "signal_timeout": 1.0,
        "heartbeat_interval": 3600,
        "trust_cache_ttl": 5,
    }
    defaults.update(overrides)
    return DatOpsConfig(**defaults)


class TestDatOpsCore(unittest.TestCase):
    """Unit tests for DatOpsCore."""

    def setUp(self):
        self.config = make_config()

    def tearDown(self):
        pass

    # ========================================================================
    # Initialization
    # ========================================================================

    @responses.activate
    def test_initialize_registers_and_verifies(self):
        """Full initialization flow: register → init reputation → verify → refresh trust."""
        # Step 1: Register agent
        responses.add(
            responses.POST,
            "http://localhost:3015/api/v1/agents",
            json={
                "success": True,
                "data": {
                    "agentDid": "did:dat:testnet:agent_test123",
                    "apiKey": "key_123",
                },
            },
            status=200,
        )

        # Step 2: Initialize reputation
        responses.add(
            responses.POST,
            "http://localhost:3015/api/v1/reputation/did:dat:testnet:agent_test123/initialize",
            json={"success": True},
            status=200,
        )

        # Step 3: Self-verify
        responses.add(
            responses.POST,
            "http://localhost:3015/api/v1/verification/agents/did:dat:testnet:agent_test123/verify-direct",
            json={"success": True},
            status=200,
        )

        # Step 4: Report verification signal
        responses.add(
            responses.POST,
            "http://localhost:3015/api/v1/reputation/did:dat:testnet:agent_test123/update",
            json={"trustScore": 55.0},
            status=200,
        )

        # Step 5: Refresh trust score
        responses.add(
            responses.GET,
            "http://localhost:3015/api/v1/reputation/did:dat:testnet:agent_test123",
            json={"data": {"trustScore": 55.5, "reliabilityScore": 55}},
            status=200,
        )

        # Step 6: Heartbeat (from heartbeat worker start, delayed)
        responses.add(
            responses.POST,
            "http://localhost:3015/api/v1/reputation/did:dat:testnet:agent_test123/heartbeat",
            json={},
            status=200,
        )

        core = DatOpsCore(self.config)
        identity = core.initialize()

        assert identity.did == "did:dat:testnet:agent_test123"
        assert identity.api_key == "key_123"
        assert core.is_ready
        assert core.trust_score == 55.5
        assert core.sandbox_level == SandboxLevel.ADAPTIVE

        core.shutdown()

    @responses.activate
    def test_initialize_already_initialized(self):
        """Second initialize() call returns cached identity."""
        # Register
        responses.add(
            responses.POST,
            "http://localhost:3015/api/v1/agents",
            json={
                "success": True,
                "data": {"agentDid": "did:dat:test:123", "apiKey": "k"},
            },
        )
        responses.add(responses.POST, "http://localhost:3015/api/v1/reputation/did:dat:test:123/initialize", json={})
        responses.add(responses.POST, "http://localhost:3015/api/v1/verification/agents/did:dat:test:123/verify-direct", json={"success": True})
        responses.add(responses.POST, "http://localhost:3015/api/v1/reputation/did:dat:test:123/update", json={"trustScore": 50})
        responses.add(responses.GET, "http://localhost:3015/api/v1/reputation/did:dat:test:123", json={"data": {"trustScore": 50}})
        responses.add(responses.POST, "http://localhost:3015/api/v1/reputation/did:dat:test:123/heartbeat", json={})

        core = DatOpsCore(self.config)
        id1 = core.initialize()
        id2 = core.initialize()

        assert id1 is id2
        core.shutdown()

    @responses.activate
    def test_registration_failure_raises(self):
        """All retry attempts failing raises RegistrationError."""
        responses.add(
            responses.POST,
            "http://localhost:3015/api/v1/agents",
            json={"success": False, "error": "Server error"},
            status=500,
        )
        responses.add(
            responses.POST,
            "http://localhost:3015/api/v1/agents",
            json={"success": False, "error": "Server error"},
            status=500,
        )

        core = DatOpsCore(self.config)
        with self.assertRaises(RegistrationError) as ctx:
            core.initialize()
        assert "registration attempts failed" in str(ctx.exception).lower()
        core.shutdown()

    def test_no_auto_register_no_identity_raises(self):
        """auto_register=False with no persisted identity raises."""
        config = make_config(auto_register=False)
        core = DatOpsCore(config)
        with self.assertRaises(RegistrationError):
            core.initialize()

    # ========================================================================
    # Trust Score
    # ========================================================================

    @responses.activate
    def test_trust_score_cache(self):
        """Trust score is cached and not re-fetched within TTL."""
        # Seed with identity
        config = make_config(trust_cache_ttl=10)
        core = DatOpsCore(config)
        core._identity = AgentIdentity(
            did="did:dat:test:cached",
            api_key="k",
            trust_score=60.0,
            sandbox_level=SandboxLevel.ADAPTIVE,
        )
        core._initialized = True

        # First fetch
        responses.add(
            responses.GET,
            "http://localhost:3015/api/v1/reputation/did:dat:test:cached",
            json={"data": {"trustScore": 65.0}},
        )

        result1 = core.get_trust_score()
        assert result1.trust_score == 65.0

        # Second fetch should use cache (no new HTTP response added)
        result2 = core.get_trust_score()
        assert result2.trust_score == 65.0
        assert len(responses.calls) == 1  # Only one HTTP call

    @responses.activate
    def test_trust_score_force_refresh(self):
        """force_refresh=True bypasses cache."""
        core = DatOpsCore(self.config)
        core._identity = AgentIdentity(
            did="did:dat:test:fr",
            api_key="k",
            trust_score=50.0,
            sandbox_level=SandboxLevel.ADAPTIVE,
        )
        core._initialized = True

        responses.add(
            responses.GET,
            "http://localhost:3015/api/v1/reputation/did:dat:test:fr",
            json={"data": {"trustScore": 70.0}},
        )
        responses.add(
            responses.GET,
            "http://localhost:3015/api/v1/reputation/did:dat:test:fr",
            json={"data": {"trustScore": 75.0}},
        )

        result1 = core.get_trust_score(force_refresh=True)
        result2 = core.get_trust_score(force_refresh=True)

        assert result1.trust_score == 70.0
        assert result2.trust_score == 75.0
        assert len(responses.calls) == 2

    def test_trust_score_no_identity_returns_zero(self):
        """Trust score with no identity returns 0 (STRICT)."""
        core = DatOpsCore(self.config)
        result = core.get_trust_score()
        assert result.trust_score == 0
        assert result.sandbox_level == SandboxLevel.STRICT

    # ========================================================================
    # Signal Reporting
    # ========================================================================

    @responses.activate
    def test_report_signal_success(self):
        """Signal reporting sends correct payload."""
        core = DatOpsCore(self.config)
        core._identity = AgentIdentity(
            did="did:dat:test:sig",
            api_key="k",
            trust_score=50.0,
            sandbox_level=SandboxLevel.ADAPTIVE,
        )

        responses.add(
            responses.POST,
            "http://localhost:3015/api/v1/reputation/did:dat:test:sig/update",
            json={"trustScore": 52.0},
        )

        core.report_signal(
            SignalEvent.ACTION_SUCCESS,
            response_time_ms=150.0,
            details={"action": "test_ok", "tool": "test"},
        )

        assert len(responses.calls) == 1
        body = json.loads(responses.calls[0].request.body)
        assert body["event"] == "action_success"
        assert body["responseTime"] == 150.0
        assert body["details"]["action"] == "test_ok"

        # Trust score updated from response
        assert core.trust_score == 52.0

    @responses.activate
    def test_report_signal_never_raises(self):
        """Signal reporting never raises (fire-and-forget)."""
        core = DatOpsCore(self.config)
        core._identity = AgentIdentity(
            did="did:dat:test:noerr",
            api_key="k",
            trust_score=50.0,
            sandbox_level=SandboxLevel.ADAPTIVE,
        )

        responses.add(
            responses.POST,
            "http://localhost:3015/api/v1/reputation/did:dat:test:noerr/update",
            body=ConnectionError("Network down"),
        )

        # Should not raise
        core.report_signal(SignalEvent.ACTION_FAILURE)

    def test_report_signal_offline_agent_skips(self):
        """Signal reporting skips for offline agents."""
        core = DatOpsCore(self.config)
        core._identity = AgentIdentity(
            did="did:dat:test:offline_agent",
            api_key="k",
        )

        # Should not make any HTTP call
        core.report_signal(SignalEvent.ACTION_SUCCESS)

    # ========================================================================
    # Identity Persistence
    # ========================================================================

    def test_persist_and_load_identity(self):
        """Identity can be persisted to file and reloaded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            config = make_config(persist_identity=path)
            core = DatOpsCore(config)
            core._identity = AgentIdentity(
                did="did:dat:test:persist",
                api_key="secret_key_123",
                trust_score=75.0,
                verification_status="verified",
                registered_at="2026-03-01T00:00:00Z",
            )

            core._persist_identity()

            # Verify file exists and has correct permissions
            assert os.path.exists(path)
            stat = os.stat(path)
            assert (stat.st_mode & 0o777) == 0o600

            # Load it back
            loaded = core._load_persisted_identity()
            assert loaded is not None
            assert loaded.did == "did:dat:test:persist"
            assert loaded.api_key == "secret_key_123"
            assert loaded.trust_score == 75.0
        finally:
            os.unlink(path)

    def test_load_nonexistent_identity_returns_none(self):
        """Loading from nonexistent file returns None."""
        config = make_config(persist_identity="/tmp/nonexistent_datops_test.json")
        core = DatOpsCore(config)
        assert core._load_persisted_identity() is None

    # ========================================================================
    # Properties
    # ========================================================================

    def test_properties_no_identity(self):
        """Properties return safe defaults when not initialized."""
        core = DatOpsCore(self.config)
        assert core.did == "not-registered"
        assert core.trust_score == 0.0
        assert core.sandbox_level == SandboxLevel.STRICT
        assert core.is_ready is False
        assert core.identity is None

    # ========================================================================
    # Response Time Clamping
    # ========================================================================

    @responses.activate
    def test_response_time_clamped(self):
        """Response time is clamped to [1, 60000]."""
        core = DatOpsCore(self.config)
        core._identity = AgentIdentity(
            did="did:dat:test:clamp",
            api_key="k",
            trust_score=50.0,
            sandbox_level=SandboxLevel.ADAPTIVE,
        )

        responses.add(
            responses.POST,
            "http://localhost:3015/api/v1/reputation/did:dat:test:clamp/update",
            json={"trustScore": 50},
        )

        core.report_signal(SignalEvent.ACTION_SUCCESS, response_time_ms=-100)

        body = json.loads(responses.calls[0].request.body)
        assert body["responseTime"] == 1  # Clamped to minimum


class TestTrustCache(unittest.TestCase):
    """Tests for the thread-safe TTL cache."""

    def test_set_and_get(self):
        from datops_agent.cache import TrustCache

        cache = TrustCache(default_ttl=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_expired_entry_returns_none(self):
        from datops_agent.cache import TrustCache

        cache = TrustCache(default_ttl=60)
        cache.set("key1", "value1", ttl=0)  # Immediate expiry
        time.sleep(0.01)
        assert cache.get("key1") is None

    def test_delete(self):
        from datops_agent.cache import TrustCache

        cache = TrustCache(default_ttl=60)
        cache.set("key1", "value1")
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_clear(self):
        from datops_agent.cache import TrustCache

        cache = TrustCache(default_ttl=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_missing_key_returns_none(self):
        from datops_agent.cache import TrustCache

        cache = TrustCache()
        assert cache.get("nonexistent") is None


class TestHeartbeat(unittest.TestCase):
    """Tests for the heartbeat worker."""

    def test_start_and_stop(self):
        from datops_agent.heartbeat import HeartbeatWorker

        called = []

        def heartbeat_fn():
            called.append(True)

        worker = HeartbeatWorker(heartbeat_fn, interval=1)
        worker.start()
        assert worker.is_alive
        worker.stop()
        assert not worker.is_alive

    def test_heartbeat_exception_swallowed(self):
        from datops_agent.heartbeat import HeartbeatWorker

        def failing_fn():
            raise RuntimeError("Network error")

        worker = HeartbeatWorker(failing_fn, interval=1)
        worker.start()
        time.sleep(0.1)
        assert worker.is_alive  # Still running despite exception
        worker.stop()


if __name__ == "__main__":
    unittest.main()
