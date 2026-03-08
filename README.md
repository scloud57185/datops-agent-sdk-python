# datops-agent-sdk

Drop-in trust enforcement for AI agent frameworks — LangChain, CrewAI, OpenAI, and more.

Add DatOps trust-gated execution to your existing AI agent in **2 lines of code**. Every tool call is authorized against the agent's live trust score, sandboxed by risk level, and reported as a trust signal — no Docker, no Redis, no infrastructure changes.

> **[Try it live in the browser](https://stackblitz.com/github/scloud57185/datops-agent-sdk-demo)** — drag the trust slider and watch tools get blocked in real-time. No install required.

## Install

```bash
pip install datops-agent-sdk
```

## Getting Your API Key

1. **Sign up** at [datops.ai/pages/signup.html](https://www.datops.ai/pages/signup.html)
   — choose Individual (personal workspace) or Organization
2. **Log in** at [datops.ai/pages/login.html](https://www.datops.ai/pages/login.html)
3. Go to **SDK Keys** in the dashboard sidebar
4. Click **Generate New Key** — give it a name, copy the key (shown once)
5. Use the key in your code or set it as an environment variable:
   ```bash
   export DAT_API_KEY="dat_xxx"
   ```

## Trust Shield Badge

Show your agent's live trust score with a clickable badge that links to your public profile on the [DAT Trust Registry](https://www.datops.ai/pages/registry.html):

```markdown
[![DAT Trust](https://www.datops.ai/api/v1/badge/<your-agent-did>.svg)](https://www.datops.ai/pages/registry?agent=<your-agent-did>)
```

Options: `?style=flat|plastic` and `?label=Custom+Label`

Your agent DID and ready-to-paste badge markdown are shown on the **SDK Keys** dashboard page after generating a key.

With framework extras:

```bash
pip install datops-agent-sdk[langchain]   # LangChain support
pip install datops-agent-sdk[crewai]      # CrewAI support
pip install datops-agent-sdk[openai]      # OpenAI Agents SDK support
pip install datops-agent-sdk[all]         # All frameworks
```

## Quick Start

### LangChain

```python
from datops_agent import DatOps

agent = DatOps.wrap_langchain(my_agent, api_key="dat_xxx")
agent.invoke({"input": "search for flights to NYC"})
```

### CrewAI

```python
from datops_agent import DatOps

crew = DatOps.wrap_crewai(my_crew, api_key="dat_xxx")
crew.kickoff()
```

### OpenAI Agents SDK

```python
from datops_agent import DatOps

agent = DatOps.wrap_openai(my_agent, api_key="dat_xxx")
```

### Generic (any framework)

```python
from datops_agent import DatOps

datops = DatOps(api_key="dat_xxx")

@datops.trust_gate(risk_level="medium")
def search_web(query: str) -> str:
    return requests.get(f"https://api.search.com?q={query}").text

# Tool call is now trust-gated
result = search_web("weather in NYC")
```

## How It Works

```
Your Agent Code
      │
      ▼
┌─────────────┐
│  DatOps SDK  │  ← 2 lines of code
├─────────────┤
│  Pre-check   │  Is this tool allowed at this trust level?
│  Execute     │  Run the tool
│  Post-report │  Report success/failure as trust signal
└─────────────┘
      │
      ▼
  DatOps Platform (trust score, reputation, sandbox level)
```

**Before each tool call:**
- Fetch the agent's trust score (cached, 60s TTL)
- Map score to sandbox level: **STRICT** (0-30), **ADAPTIVE** (30-70), **OPEN** (70-100)
- Check if the tool's risk level is allowed in the current sandbox
- Block execution if trust is too low

**After each tool call:**
- Report success or failure as a trust signal (fire-and-forget)
- Signals feed back into the agent's reputation score

## Sandbox Levels

| Trust Score | Sandbox    | Allowed Risk Levels |
|-------------|------------|---------------------|
| 0 - 30      | STRICT     | Low only            |
| 30 - 70     | ADAPTIVE   | Low + Medium        |
| 70 - 100    | OPEN       | All (Low/Medium/High) |

New agents start at trust score **50** (ADAPTIVE). As the agent demonstrates reliability, its trust grows and more tools become available.

## Configuration

```python
datops = DatOps(
    api_key="dat_xxx",                          # Required
    base_url="https://www.datops.ai",           # Platform URL
    agent_name="my-agent",                      # Display name
    network="testnet",                          # testnet | mainnet
    trust_cache_ttl=60,                         # Cache trust score (seconds)
    heartbeat_interval=300,                     # Heartbeat interval (seconds)
    min_trust_for_tool=10.0,                    # Minimum trust to use any tool
    trust_threshold_high_risk=70.0,             # Minimum trust for high-risk tools
    persist_identity="~/.datops/identity.json", # Persist agent DID across restarts
    auto_initialize=True,                       # Auto-register on first use
    debug=False,                                # Enable debug logging
)
```

## Risk Levels

Assign risk levels to control which sandbox levels can execute each tool:

```python
@datops.trust_gate(risk_level="low")
def read_file(path: str) -> str:
    """Low risk — available in all sandbox levels."""
    ...

@datops.trust_gate(risk_level="medium")
def search_web(query: str) -> str:
    """Medium risk — requires ADAPTIVE or OPEN sandbox."""
    ...

@datops.trust_gate(risk_level="high")
def send_email(to: str, body: str) -> str:
    """High risk — requires OPEN sandbox (trust >= 70)."""
    ...
```

For framework adapters, set per-tool risk levels:

```python
# CrewAI
crew = DatOps.wrap_crewai(
    my_crew,
    api_key="dat_xxx",
    tool_risk_levels={
        "search_tool": "low",
        "email_tool": "high",
    },
)

# OpenAI
agent = DatOps.wrap_openai(
    my_agent,
    api_key="dat_xxx",
    tool_risk_levels={
        "web_search": "medium",
        "send_email": "high",
    },
)
```

## Error Handling

```python
from datops_agent import DatOps, ToolBlockedError

datops = DatOps(api_key="dat_xxx")

@datops.trust_gate(risk_level="high")
def dangerous_tool():
    ...

try:
    dangerous_tool()
except ToolBlockedError as e:
    print(f"Blocked: {e.reason}")
    print(f"Trust: {e.trust_score}, Sandbox: {e.sandbox_level}")
```

## Inspecting Trust State

```python
datops = DatOps(api_key="dat_xxx")

# Current trust score
print(datops.trust_score)  # 55.0

# Sandbox info
info = datops.get_sandbox_info()
print(info)
# {'trust_score': 55.0, 'sandbox_level': 'ADAPTIVE', 'allowed_risk_levels': ['low', 'medium']}

# Agent DID
print(datops.did)  # did:dat:testnet:agent_abc123

# Force refresh
score = datops.get_trust_score(force_refresh=True)
```

## Context Manager

```python
with DatOps(api_key="dat_xxx") as datops:
    @datops.trust_gate(risk_level="medium")
    def my_tool():
        return "result"

    my_tool()
# Heartbeat stopped, resources cleaned up
```

## Identity Persistence

By default, the SDK generates a new agent DID on each startup. To persist identity across restarts:

```python
datops = DatOps(
    api_key="dat_xxx",
    persist_identity="~/.datops/identity.json",
)
```

The identity file stores the agent's DID, API key, and trust state. It's created with `chmod 600` (owner read/write only).

## Architecture

```
datops_agent/
  __init__.py          # DatOps class (public API)
  core.py              # Registration, trust cache, signal reporting
  trust_gate.py        # Pre/post tool call middleware
  cache.py             # Thread-safe TTL cache (no Redis)
  heartbeat.py         # Background daemon thread
  types.py             # Enums, dataclasses, exceptions
  adapters/
    langchain.py       # LangChain CallbackHandler
    crewai.py          # CrewAI tool wrapping
    openai_sdk.py      # OpenAI SDK tool wrapping
    generic.py         # Decorator pattern
```

**No Redis. No Docker. No infrastructure.** Pure Python with `requests` as the only dependency.

## Development

```bash
git clone https://github.com/datops-ai/agent-sdk-python.git
cd agent-sdk-python
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
