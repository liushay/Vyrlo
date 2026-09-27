# Vyrlo

A composable Agent runtime framework — five-layer pluggable architecture, middleware-driven, local-first.

## Core Features

- **Five-layer Pluggable Architecture**: Agent core → Assembly layer → Middleware layer → Multi-Agent orchestration → Skill evolution; each layer independently replaceable
- **15 Built-in Middlewares**: Memory management, safety guards, observability, sandbox isolation, circuit breaker/retry, skill injection, ready out of the box
- **Local-first Deployment**: Only PyYAML + SQLite required; the full environment runs without external services
- **Extensible Tools**: Unified ToolRegistry interface — register custom tools with a single decorator
- **Triple-channel Observability**: Prometheus metrics + JSONL event log + Tracing spans; fault localization in minutes
- **One-click Docker Deployment**: `docker compose up` to start with health checks, metrics endpoint, and Grafana dashboard

## Architecture Overview

```
┌──────────────────────────────────────────────┐
│  L5: Skill Evolution System                   │
│  SkillSystem = Store + Selector + Mutator      │
│  + Evaluator + Monitor                         │
├──────────────────────────────────────────────┤
│  L4: Multi-Agent Orchestration                │
│  AgentRegistry + SharedState + Delegation      │
│  + Orchestrator                                │
├──────────────────────────────────────────────┤
│  L3: Middleware Layer (15 built-in)            │
│  Memory / Safety / Observability / Isolation   │
│  / Resilience / Skill Injection                │
├──────────────────────────────────────────────┤
│  L2: Four Core Components + Assembly           │
│  LLMAdapter + ToolRegistry + ContextManager     │
│  + ToolCallParser → Runtime Assembly           │
├──────────────────────────────────────────────┤
│  L1: Agent Core                                │
│  AgentLoop + Context + Middleware              │
└──────────────────────────────────────────────┘
```

## Quick Start

### Requirements

- Python 3.10+
- (Optional) Ollama — if you want to use local models

### Install & Run

```bash
# Clone the repo
git clone https://github.com/liushay/Vyrlo.git
cd Vyrlo

# Install dependencies (only PyYAML is mandatory)
pip install -r requirements.txt

# Start the service (uses echo provider, no API key needed)
python serve.py

# Verify
curl http://localhost:8000/health
```

Expected output:

```json
{"status": "ok", "components": {"llm": "healthy", "sandbox": "healthy", "storage": "healthy"}}
```

### Run a Minimal Task

```python
from runtime import Runtime
from runtime.llm_adapter.multi_provider import EchoProvider, MultiProviderAdapter
from runtime.llm_adapter.interface import ModelConfig

# Echo provider always echoes user input — no API key required
adapter = MultiProviderAdapter(
    default_config=ModelConfig(provider="echo", model="echo-test")
)
adapter.register_provider(EchoProvider())

rt = Runtime(llm_adapter=adapter)
result = rt.run(messages=[{"role": "user", "content": "Hello, Vyrlo!"}])
print(result["assistant"])
```

## Configuration

Configuration file is at `config/default.yaml`, with environment variable override support (`APP_` prefix + double-underscore nesting).

```bash
# Use a custom config file
python serve.py --config /path/to/config.yaml

# Override LLM provider via environment variable
APP_LLM__PROVIDER=openai python serve.py
```

Key configuration items:

| Config Path | Description | Default |
|-------------|-------------|---------|
| `llm.provider` | LLM provider | `echo` |
| `sandbox.mode` | Sandbox mode | `local` |
| `storage.backend` | Storage backend | `sqlite` |
| `server.port` | HTTP server port | `8000` |

See `config/schema.json` for the full configuration schema.

## Extending Vyrlo

### Adding a Middleware

Subclass `Middleware` and override the lifecycle hooks you need:

```python
from runtime import Middleware, HookResult

class LoggingMiddleware(Middleware):
    async def before_llm_call(self, ctx, messages):
        print(f"[log] sending {len(messages)} messages to LLM")
        return HookResult.CONTINUE

# Register at runtime
rt.register_middleware(LoggingMiddleware())
```

Supported hooks: `before_iteration`, `after_iteration`, `before_llm_call`, `after_llm_call`, `before_tool`, `after_tool`, `on_error`.

### Adding a Tool

Register any function as a tool with a single decorator:

```python
from runtime.tool_registry import tool_registry

@tool_registry.register(name="weather", description="Get weather for a city")
def get_weather(city: str) -> dict:
    return {"city": city, "temperature": 22, "condition": "sunny"}
```

### Adding a Skill

The skill system supports dynamic loading, mutation, and evaluation. See `docs/USER_GUIDE.md` §4 for details.

## Known Limitations

For current version limitations (model capability boundaries, single-point dependencies, tool count, etc.), see [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

## License

MIT License — see [LICENSE](LICENSE).