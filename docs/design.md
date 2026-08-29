# Framework Design

## Goals

Engram is designed around five constraints:

1. The complete control flow must be readable without learning a workflow language.
2. Provider-specific objects must stop at the model-client boundary.
3. Tools must be self-describing, validated, observable, and usable from sync or async code.
4. Conversation, retrieved context, durable memory, and external knowledge must remain distinct.
5. Advanced features must be replaceable through small protocols rather than global configuration.

The result is a layered framework rather than a single feature-rich base class.

## Layers

### Model layer

`ModelClient` is the dependency-inversion boundary used by every agent. It has two operations:
`complete()` and `acomplete()`. Both return `ModelResponse`, which normalizes text, tool calls,
continuation items, and token usage.

`LLMClient` implements the protocol with the Responses API. The normalized result prevents agent
code from importing provider response classes. `output_items` are retained because a correct
stateless function-calling continuation must send model output items back together with
`function_call_output` items.

The framework does not infer named providers. A model name, key, and endpoint are explicit inputs
or environment variables. Any compatible local or hosted service can therefore be selected
without adding provider branches to the framework.

### Message and context layer

`Message` stores a role, content, timestamp, and application metadata. It deliberately represents
conversation messages only. Tool calls and tool outputs are transient run items, not fake chat
roles.

`HistoryManager` owns bounded conversation history and can select the newest messages that fit a
token budget. It also supports explicit compaction: an application or model creates a summary, and
the manager replaces old turns with that summary while retaining recent messages.

`ContextBuilder` follows a gather-select-structure flow:

1. Gather conversation history and application-supplied `ContextSource` objects.
2. Select high-priority sources within a dedicated fraction of the input budget.
3. Select recent history with the remaining budget.
4. Structure retrieved sources as a developer message followed by conversation messages.

The default counter is a conservative dependency-free estimate. Applications that know their
model tokenizer can provide another `TokenCounter`.

### Tool layer

`Tool` combines a callable, name, description, and JSON Schema. Callable annotations are converted
to schema and validated with Pydantic before execution. Explicit schemas support remotely
discovered tools.

`ToolResult` uses three statuses:

- `success`: the operation completed normally;
- `partial`: usable output exists but is incomplete or degraded;
- `error`: no valid output was produced.

The model receives a JSON representation containing status, display content, structured data,
error details, and metadata. Applications can inspect the same object without parsing prose.

`ToolRegistry` owns uniqueness, discovery, safe execution, timing, parallel calls, and conversion
of validation or runtime exceptions into `ToolResult`. `ToolPipeline` is intentionally explicit:
each step names a tool and supplies arguments or an argument builder based on the prior result.

### Agent layer

`Agent` owns only shared identity and infrastructure: model, instructions, configuration, tools,
history, context builder, events, and session helpers. It does not implement a universal run loop.

Concrete patterns are small:

- `ToolAgent` runs the native function-calling loop.
- `SimpleAgent` supplies general conversational instructions to `ToolAgent`.
- `ReActAgent` supplies instructions for deliberate reason-act-observe behavior to the same loop.
- `PlanAndSolveAgent` validates a JSON plan and executes every step with accumulated results.
- `ReflectionAgent` generates, critiques, and refines a result while retaining a typed trajectory.

`agent_as_tool()` is the composition primitive. A coordinator can expose focused agents as tools,
which gives delegation the same schema, error, and observability behavior as any other capability.
No separate multi-agent runtime is necessary for local coordination.

## Native Tool-Calling Flow

For one `ToolAgent.run()` call:

1. Validate and append the user message.
2. Build token-bounded input from history and optional retrieved sources.
3. Send input, instructions, and tool schemas to the model.
4. If the model returns text without tool calls, persist and return it.
5. Otherwise, append every model output item to the transient input.
6. Validate and execute requested tools, optionally in parallel.
7. Append one `function_call_output` item per call.
8. Repeat until the model returns final text or the configured step limit is reached.

Only the user and final assistant messages enter durable conversation history. Detailed tool
activity is emitted as events and can be recorded separately.

## Memory and Retrieval

Durable agent memory and source retrieval solve different problems and have separate modules.

`MemoryManager` operates on `MemoryStore`, whose minimal contract is add, get, remove, and list.
Each `Memory` has a cognitive kind, importance, tags, metadata, and creation time. Default recall
combines lexical relevance, importance, and recency. `InMemoryStore` is the zero-configuration
backend; `JsonMemoryStore` is an atomic local backend. Database adapters can implement the same
protocol.

`KnowledgeBase` ingests `Document` objects, splits them into traceable `Chunk` objects, and indexes
them through a `SearchIndex`. `LexicalIndex` is deterministic and dependency-free. `VectorIndex`
accepts any `EmbeddingModel`, so hosted embedding services and local models remain integrations.
Search results preserve chunk and document identifiers for citation or provenance.

Both systems can become model tools. Retrieved knowledge can also be converted directly into a
`ContextSource` when retrieval is controlled by application code rather than by the model.

## Protocol Integration

MCP already defines transports and client-session lifetime, so Engram does not duplicate them.
`MCPToolProvider` consumes the small `MCPSession` protocol, discovers remote schemas, and creates
async Engram tools. The application owns authentication, connection setup, retries, and shutdown.

Remote agent delegation does not require a framework-specific wire protocol. A remote transport
can be wrapped as a normal tool, while local agents use `agent_as_tool()`. This avoids binding the
core to protocol specifications that evolve independently.

## Sessions and Events

`JsonSessionStore` persists a versioned session containing agent identity, conversation messages,
metadata, and update time. File names are constrained and writes use an atomic replace.

`EventBus` emits run, model, and tool lifecycle events. Handlers decide whether to log, collect
metrics, stream UI updates, or ignore them. `TraceRecorder` is one handler that keeps sanitized
events in memory and writes JSON Lines only on request. Credential-like keys are redacted before
storage.

## Evaluation

Evaluation is runner-oriented rather than agent-class-oriented. `Evaluator` accepts any
`Callable[[str], str]`, a sequence of `EvaluationCase` objects, and named metric callables. It
returns per-case results plus aggregate averages and error counts.

The built-in deterministic metrics cover exact and containment checks. `LLMJudge` implements the
same metric protocol for rubric-based scoring. Benchmark loaders, human-review interfaces, and
leaderboard exporters belong in separate packages because they carry large, fast-changing data
and dependency surfaces.

## Intentional Redesigns

Several tempting designs are deliberately avoided:

- No provider guessing from key prefixes or URLs. Explicit configuration is predictable.
- No giant configuration model. Each component owns its small settings surface.
- No model-generated Python list parsing. Plans use JSON.
- No text-regex action parser for tool-capable models. Native function calls are structured.
- No automatic hidden summary model. History compaction is explicit and testable.
- No mandatory vector database or graph database. Storage and embeddings are protocols.
- No automatic file, terminal, or network mutation tools. Applications opt into capabilities.
- No training stack in the runtime package. Training and inference have incompatible dependency
  and operational profiles.
- No benchmark-specific classes in the core. Evaluation primitives remain reusable.

These boundaries keep the framework complete for agent applications while preserving the ability
to understand and replace every component.
