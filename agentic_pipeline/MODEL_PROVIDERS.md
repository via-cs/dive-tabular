# Model providers

The categorical, equational, and linear agentic constraint-discovery pipelines
support native OpenAI and Anthropic APIs.

## Provider selection

OpenAI remains the default:

```bash
--provider openai --model gpt-5.6-luna
```

Use Claude Sonnet 5 with:

```bash
--provider anthropic --model claude-sonnet-5
```

Live calls load `.env` and require the key for the selected provider:

```text
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

`--max-output-tokens` sets Anthropic's required per-turn output limit and
defaults to `32768`. OpenAI calls preserve their previous behavior and do not
send this Anthropic-specific parameter.

For example, run equational discovery with Claude:

```bash
uv run python -m agentic_pipeline.equational \
  dataset/nba dataset/nba \
  --provider anthropic \
  --model claude-sonnet-5 \
  --max-constraints 20 \
  --max-discovery-phases 2 \
  --max-refinement-rounds 3 \
  --skip-consolidation \
  --skip-fix-generation
```

Run linear discovery with Claude:

```bash
uv run python -m agentic_pipeline.linear \
  dataset/url dataset/url \
  --provider anthropic \
  --model claude-sonnet-5 \
  --max-constraints 20 \
  --max-discovery-phases 2 \
  --max-refinement-rounds 3 \
  --skip-consolidation
```

## Backend boundary

`model_backends.py` normalizes text, usage, and client-side tool calls while
using each provider's native API:

- OpenAI uses the Responses API.
- Anthropic uses the streaming Messages API, assembles each final message, and
  supports `tool_use`/`tool_result`, strict tool schemas, and
  `output_config.format`. Streaming avoids the SDK's ten-minute safeguard for
  high output-token budgets.
- Anthropic's SDK transforms strict Pydantic schemas to its supported JSON
  Schema subset. Unsupported constraints remain in field descriptions and are
  still enforced by the original host-side Pydantic models. Non-strict dynamic
  schemas, such as linear coefficient maps keyed by column name, are preserved.
- Raw assistant content is retained between turns. This preserves signed
  Anthropic thinking blocks when a verifier conversation continues.
- Parallel tool use is disabled because each discovery or refinement turn is
  designed to submit one batched verifier call.

The full-data verifier, generated-code sandboxing, refinement policy,
constraint artifacts, and evaluation metrics are unchanged by provider
selection.
